import logging
import threading
import time
from queue import Empty, Full, Queue

from lerobot.async_inference.helpers import RawObservation, TimedObservation
from lerobot.async_inference.robot_client import RobotClient


class AsyncObservationRobotClient(RobotClient):
    """RobotClient variant that sends observations in a background thread.

    The control loop only enqueues observations, so network delays in gRPC do not
    block action execution on the main thread.
    """

    def __init__(
        self,
        config,
        observation_queue_size: int = 1,
        sender_join_timeout_s: float = 3.0,
    ):
        super().__init__(config)

        self._obs_send_queue: Queue = Queue(maxsize=observation_queue_size)
        self._obs_sender_stop_event = threading.Event()
        self._obs_sender_thread: threading.Thread | None = None

        self._sender_join_timeout_s = sender_join_timeout_s

        # args for ensure all observations are sent
        self._obs_in_flight_count: int = 0
        self._obs_in_flight_lock = threading.Lock()

    def clear_pending_observations(self):
        while True:
            try:
                self._obs_send_queue.get_nowait()
            except Empty:
                break

    def start(self):
        ok = super().start()
        if not ok:
            raise RuntimeError("Failed to start RobotClient")

        self._obs_sender_stop_event.clear()
        self._obs_sender_thread = threading.Thread(
            target=self._observation_sender_loop,
            daemon=True,
        )
        self._obs_sender_thread.start()
        self.logger.info("Background observation sender thread started")

    def stop(self):
        self._obs_sender_stop_event.set()

        if self._obs_sender_thread and self._obs_sender_thread.is_alive():
            self._obs_sender_thread.join(timeout=self._sender_join_timeout_s)

        super().stop()

    def send_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation for background sending.

        Returns True when enqueued successfully, False when dropped.
        """
        if not self.running:
            raise RuntimeError(
                "Client not running. Run RobotClient.start() before sending observations."
            )

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        try:
            self._obs_send_queue.put_nowait(obs)
            return True
        except Full:
            try:
                _ = self._obs_send_queue.get_nowait()
            except Empty:
                pass

            try:
                self._obs_send_queue.put_nowait(obs)
                return True
            except Full:
                return False

    def _observation_sender_loop(self) -> None:
        while not self._obs_sender_stop_event.is_set():
            try:
                observation = self._obs_send_queue.get(timeout=0.05)
                with self._obs_in_flight_lock:
                    self._obs_in_flight_count += 1
            except Empty:
                continue

            send_start = time.perf_counter()
            try:
                super().send_observation(observation)
            finally:
                self.logger.debug(
                    "Background observation send latency: "
                    f"{(time.perf_counter() - send_start) * 1000:.2f}ms | "
                    f"pending={self._obs_send_queue.qsize()}"
                )
                with self._obs_in_flight_lock:
                    self._obs_in_flight_count -= 1

    def wait_for_all_observations_sent(self):
        logging.info("Waiting for all pending observations to be sent...")
        while True:
            with self._obs_in_flight_lock:
                in_flight = self._obs_in_flight_count
            if self._obs_send_queue.empty() and in_flight == 0:
                break
            time.sleep(0.5)

    # override
    def control_loop_observation(self, task: str, verbose: bool = False):
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                # change and to or
                observation.must_go = self.must_go.is_set() or self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            _ = self.send_observation(observation)

            self.logger.debug(
                f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})"
            )
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(
                    observation.get_timestamp()
                )

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")
