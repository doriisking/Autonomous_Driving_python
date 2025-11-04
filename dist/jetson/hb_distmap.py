#돌아가는 코드 -> 엄청 느림

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.path.dirname(__file__))

import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker
# 파일 상단에 추가
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import torch

# 거리맵 함수 (같은 폴더의 distmap_def.py)
from distmap_cuda_def import (
    build_dist_map_bfs_cuda,
    build_dist_map_bf_cuda,
    #build_dist_map_bruteforce,
    distmap_to_occupancygrid,  # 필요 시 사용
)


class DWACommandNode(Node):
    """
    전방 창(window)에서 최소 코스트 셀을 골라 /cmd(Twist) 발행.
    - 코스트: (x-dx)^2 + (y-dy)^2 + [d<margin] * penalty * (1 - d/margin)^2
    - vx(+전진), vyaw(+좌회전) 생성
    - 옵션(현재 비활성): EMA 필터, 가속 한계
    """

    def __init__(self):
        super().__init__("dwa_command_node")

        # -------------------- 기본/코스트 파라미터 --------------------
        self.declare_parameter("penalty", 10.0)            # 장애물 페널티 상수
        self.declare_parameter("margin", 0.3)              # 안전 여유[m]
        self.declare_parameter("dx", 1.5)                  # 상위(GPS)가 준 목표 x[m] (로봇 기준)
        self.declare_parameter("dy", 0.0)                  # 상위(GPS)가 준 목표 y[m] (로봇 기준)

        # -------------------- 검사 창(Window) --------------------
        self.declare_parameter("ahead_m", 2.5)             # 전방 길이[m]
        self.declare_parameter("half_width_m", 1.0)        # 좌우 반폭[m]
        self.declare_parameter("stride", 1)                # 셀 스킵 간격

        # -------------------- 속도 생성 파라미터 --------------------
        self.declare_parameter("kv", 0.6)                  # 거리→전진속도 게인
        self.declare_parameter("kyaw", 2.0)                # 각도→회전속도 게인
        self.declare_parameter("v_max", 0.6)               # 전진 최대[m/s]
        self.declare_parameter("w_max", 1.0)               # 회전 최대[rad/s]
        self.declare_parameter("v_min", 0.0)               # 전진 최소[m/s]

        # -------------------- 안전/회전 우선 --------------------
        self.declare_parameter("safety_slowdown", True)    # d<margin 감속
        self.declare_parameter("enable_turn_in_place", True)
        self.declare_parameter("theta_turn_deg", 35.0)     # 큰 각도면 제자리 회전
        self.declare_parameter("allow_backward_target", False)

        # -------------------- 주기 --------------------
        self.declare_parameter("timer_dt", 0.1)            # 타이머 주기(초)

        # -------------------- 토픽 --------------------
        self.declare_parameter("occ_topic", "/bev/occupancy_grid")
        self.declare_parameter("cmd_topic", "/cmd")
        self.declare_parameter("marker_topic", "/dwa/local_goal_marker")

        # ---- 거리맵 관련 (방식 토글 + 최대거리 + 시각화) ----
        self.declare_parameter("dist_method", "bruteforce")       # "bfs" | "bruteforce"
        self.declare_parameter("dist_max_m", 3.0)          # 거리맵 최대 반경[m]
        self.declare_parameter("publish_distgrid", False)  # 거리맵을 OccGrid로 내보내기

        # ---- 파라미터 로드 ----
        self.penalty = float(self.get_parameter("penalty").value)
        self.margin  = float(self.get_parameter("margin").value)
        self.dx      = float(self.get_parameter("dx").value)
        self.dy      = float(self.get_parameter("dy").value)

        self.ahead_m      = float(self.get_parameter("ahead_m").value)
        self.half_width_m = float(self.get_parameter("half_width_m").value)
        self.stride       = int(self.get_parameter("stride").value)

        self.kv    = float(self.get_parameter("kv").value)
        self.kyaw  = float(self.get_parameter("kyaw").value)
        self.v_max = float(self.get_parameter("v_max").value)
        self.w_max = float(self.get_parameter("w_max").value)
        self.v_min = float(self.get_parameter("v_min").value)

        self.slow           = bool(self.get_parameter("safety_slowdown").value)
        self.turn_mode      = bool(self.get_parameter("enable_turn_in_place").value)
        self.theta_turn     = math.radians(float(self.get_parameter("theta_turn_deg").value))
        self.allow_backward = bool(self.get_parameter("allow_backward_target").value)

        # (옵션) EMA/RateLimit은 비활성 기본값으로 고정
        self.use_ema = False
        self.alpha_v = 0.7
        self.alpha_w = 0.6
        self.use_rlim = False
        self.ax_limit = 0.8
        self.aw_limit = 1.5

        self.dt = float(self.get_parameter("timer_dt").value)

        self.occ_topic    = self.get_parameter("occ_topic").value
        self.cmd_topic    = self.get_parameter("cmd_topic").value
        self.marker_topic = self.get_parameter("marker_topic").value

        self.dist_method  = str(self.get_parameter("dist_method").value).lower()
        self.dist_max_m   = float(self.get_parameter("dist_max_m").value)
        self.pub_distgrid = bool(self.get_parameter("publish_distgrid").value)

        # ---- 상태 ----
        self._occ  = None                  # OccupancyGrid data (int8 HxW)
        self._info = None                  # (res, W, H, x0, y0)
        self._dist = None                  # 거리맵 (float32 HxW) [m]
        self._vx_prev = 0.0
        self._wz_prev = 0.0
        self._t_prev  = time.time()

        # ---- I/O (파라미터 로드 이후에 생성!) ----
        self.create_subscription(OccupancyGrid, self.occ_topic, self._cb_occ, 10)
        self.pub_cmd    = self.create_publisher(Twist, self.cmd_topic, 10)
        self.pub_marker = self.create_publisher(Marker, self.marker_topic, 10)
        # (옵션) 거리맵 시각화하고 싶으면 주석 해제
        # self.pub_dist_occ = self.create_publisher(OccupancyGrid, "/dwa/dist_grid", 10)

        self.timer = self.create_timer(self.dt, self._on_timer)

        self.get_logger().info(
            f"[dwa_command_node] L={self.ahead_m}m, ±{self.half_width_m}m | "
            f"penalty={self.penalty}, margin={self.margin} | "
            f"kv={self.kv}, kyaw={self.kyaw}, vmax={self.v_max}, wmax={self.w_max} | "
            f"stride={self.stride}, dt={self.dt}s | TurnInPlace={self.turn_mode}"
        )

    # ------------------------- 콜백: OccupancyGrid -------------------------
    def _cb_occ(self, msg: OccupancyGrid):
        H = int(msg.info.height)
        W = int(msg.info.width)
        self._occ = np.asarray(msg.data, dtype=np.int8).reshape(H, W)
        self._info = (
            float(msg.info.resolution),
            W, H,
            float(msg.info.origin.position.x),
            float(msg.info.origin.position.y),
        )

        # 거리맵 생성 (BFS / brute-force)
        if self.dist_method == "bfs":
            self._dist = build_dist_map_bfs_cuda(msg, max_dist=self.dist_max_m)
        else:
            self._dist = build_dist_map_bf_cuda(msg, max_dist=self.dist_max_m)

        # (옵션) 거리맵을 OccGrid로 내보내어 RViz에서 확인
        # if self.pub_distgrid:
        #     dist_occ = distmap_to_occupancygrid(self._dist, msg, max_dist=self.dist_max_m)
        #     self.pub_dist_occ.publish(dist_occ)

    # ------------------------- 주기 처리 -------------------------
    def _on_timer(self):
        if self._occ is None or self._info is None:
            return

        # dt 추정 (필요시 self.dt = dt 로 치환)
        t_now = time.time()
        dt = max(1e-3, t_now - self._t_prev)
        self._t_prev = t_now

        res, W, H, x0, y0 = self._info

        # 로봇(0,0)의 격자 인덱스 (i: y, j: x)
        j0 = int((0.0 - x0) / res)
        i0 = int((0.0 - y0) / res)

        # 전방 창
        j_start = max(0, j0)
        j_end   = min(W, j0 + int(self.ahead_m / res) + 1)
        i_start = max(0, i0 - int(self.half_width_m / res))
        i_end   = min(H, i0 + int(self.half_width_m / res) + 1)
        if j_start >= j_end or i_start >= i_end:
            return

        # ------ 최소 코스트 셀 탐색 ------
        best = None  # (cost, i, j, x, y, d)
        m = max(1e-6, self.margin)
        step = max(1, self.stride)

        for i in range(i_start, i_end, step):
            y = i * res + y0
            base_y = (y - self.dy) ** 2
            for j in range(j_start, j_end, step):
                x = j * res + x0

                # [핵심 수식 1] 목표 추종 항
                base = (x - self.dx) ** 2 + base_y

                # 거리 값 (m)
                if self._dist is not None:
                    d = float(self._dist[i, j])
                else:
                    d = self._distance_stub(i, j)

                # [핵심 수식 2] 장애물 페널티
                obs = self.penalty * (1.0 - d / m) ** 2 if d < m else 0.0

                cost = base + obs
                if (best is None) or (cost < best[0]):
                    best = (cost, i, j, x, y, d)

        if best is None:
            return

        _, bi, bj, bx, by, bd = best
        dx_dwa, dy_dwa = bx, by  # 로컬 목표 (로봇 기준)

        # --- RViz Marker (로컬 목표) ---
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "map"
        marker.ns = "dwa_local_goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(dx_dwa)
        marker.pose.position.y = float(dy_dwa)
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.2
        marker.color.r = 1.0; marker.color.g = 0.2; marker.color.b = 0.2; marker.color.a = 1.0
        self.pub_marker.publish(marker)

        # ------ 속도 생성 ------
        theta = math.atan2(dy_dwa, dx_dwa)   # +면 좌회전
        r = math.hypot(dx_dwa, dy_dwa) 


        vx_raw = self.kv * r * math.cos(theta) # cos커지면 vx 너무 작아짐
        #vx_raw = 0.7
        wz_raw = self.kyaw * theta

        if not self.allow_backward and dx_dwa < 0.0:
            vx_raw = 0.0

        if self.turn_mode and abs(theta) > self.theta_turn:
            vx_raw = 0.0  # 제자리 회전

        if self.slow and bd < m: ## x,yaw 밸런싱
            scale = max(0.0, min(1.0, bd / m))
            vx_raw *= scale

        # (필터/레이트리밋 비활성: 바로 포화)
        vx_cmd = max(self.v_min, min(self.v_max, vx_raw))
        wz_cmd = max(-self.w_max, min(self.w_max, wz_raw))

        # 퍼블리시
        cmd = Twist()
        cmd.linear.x  = float(vx_cmd)
        cmd.angular.z = float(wz_cmd)
        self.pub_cmd.publish(cmd)

        # 디버그
        self.get_logger().info(
            f"cmd vx={cmd.linear.x:.2f} m/s, vyaw={cmd.angular.z:.2f} rad/s | "
            f"best({bx:.2f},{by:.2f}) θ={math.degrees(theta):.1f}° d={bd:.2f}"
        )

    # ------------------------- 거리 훅 (fallback) -------------------------
    def _distance_stub(self, i: int, j: int) -> float:
        # 점유(>0)면 0, 그 외 margin*2
        if self._occ is None:
            return self.margin * 2.0
        H, W = self._occ.shape
        if 0 <= i < H and 0 <= j < W and self._occ[i, j] > 0:
            return 0.0
        return self.margin * 2.0


def main(args=None):
    rclpy.init(args=args)
    node = DWACommandNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
