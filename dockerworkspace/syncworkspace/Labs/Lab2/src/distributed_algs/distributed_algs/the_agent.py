import rclpy
from rclpy.node import Node
from time import sleep
from std_msgs.msg import Float32MultiArray as MsgFloat

class TheAgent(Node):
    def __init__(self):
        super().__init__(
            "parametric_agent",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True 
        )
        self.agent_id = self.get_parameter("id").value
        self.neighbors = self.get_parameter("neighbors").value
        self.x0 = self.get_parameter("xzero").value
        self.k = 0
        self.x_i = self.x0
        
        print(f"Agent {self.agent_id} is ready")
        print(f"My neighbors are: {self.neighbors}")
        print(f"My starting state is: {self.get_parameter('xzero').value}")

        self.pub = self.create_publisher(MsgFloat, f"/topic_{self.agent_id}", 10)

        self.timer = self.create_timer(1, self.timer_callback)

        for j in self.neighbors:
            self.create_subscription(
                MsgFloat,
                f"/topic_{j}",
                self.listener_callback,
                10
            )
        
        self.received_data = {j: [] for j in self.neighbors}

    def listener_callback(self, msg):
        j = int(msg.data[0])
        msg_j = msg.data[1:]
        self.received_data[j].append(msg_j)

    def timer_callback(self):
        if self.k == 0:
            msg = MsgFloat()
            msg.data = [float(self.agent_id), float(self.k), float(self.x0)]
            self.pub.publish(msg)
            self.k += 1
        else:
            time_stamp = [
                self.k - 1 == self.received_data[j][0][0] for j in self.neighbors
            ]
            if all(time_stamp):
                loc_est_max = self.x_i
                for j in self.neighbors:
                    loc_est_max = max(loc_est_max, self.received_data[j][0][1])
                self.x_i = loc_est_max

                msg = MsgFloat()
                msg.data = [float(self.agent_id), float(self.k), float(loc_est_max)]
                self.pub.publish(msg)
                self.k += 1
                

def main():
    rclpy.init()
    anAgent = TheAgent()
    sleep(1)
    rclpy.spin(anAgent)
    anAgent.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()