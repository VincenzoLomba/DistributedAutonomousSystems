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
        self.k = 0 # time-stamp parameter, alias, actual istant of time
        
        print(f"Agent {self.agent_id} is ready")
        print(f"My neighbors are: {self.neighbors}")
        print(f"My starting state is: {self.get_parameter('xzero').value}")

        # Creating the publisher (to send data to the neighbors)
        # There is a topic for each agent, named "/topic_{agent_id}"
        # Also, the maximum number of messages that can be stored in the pub-queue is 10
        # Notice: as soon as new data is published, all currently active subscribers are notified with that data
        self.pub = self.create_publisher(MsgFloat, f"/topic_{self.agent_id}", 10)

        # Creating the timer (to make recurrent in time the agent behavior)
        self.timer = self.create_timer(1, self.timer_callback)

        # Creating the subscribers (to receive data from all the neighbors)
        # Subscribing to the topic associated to each neighbor, named "/topic_{neighbor_id}"
        # The used callback function is listener_callback
        # The maximum number of un-elaborated messages that can be stored in the sub-queue is 10
        # Notice: as soon as new data is published in the topic, the callback function of all related subscribers is called
        for j in self.neighbors:
            self.create_subscription(
                MsgFloat,
                f"/topic_{j}",
                self.listener_callback,
                10
            )
        
        # Defining the received_data dictionary:
        # keys are the neighbors indexes, values are empty (for now) lists
        self.received_data = {j: [] for j in self.neighbors}

    def listener_callback(self, msg):
        # Receiving some data (first getting sender agent id, then, saving the data)
        j = int(msg.data[0]) # Getting sender agent id
        msg_j = msg.data[1:] # A couple of data, the time-stamp and the value: [k, value]
        self.received_data[j].append(msg_j)

    def timer_callback(self):
        if self.k == 0:
            # First istant of time for the current agent (initialization code)
            msg = MsgFloat()
            msg.data = [float(self.agent_id), float(self.k), float(self.x0)]
            self.pub.publish(msg)
            self.x_i = self.x0
            self.k += 1
        else:
            # Waiting for all the neighbors to send their data in association to the same istant of time
            # self.received_data[j][0][0] is the time-stamp of the first (alias oldtest, still non popped) data received from the j-th neighbor
            # self.received_data[j][0][1] is the related value
            time_stamps = [
                self.k - 1 == self.received_data[j][0][0] for j in self.neighbors
            ]
            if all(time_stamps):
                loc_est_max = self.x_i
                for j in self.neighbors:
                    (_, x_j) = self.received_data[j].pop(0)
                    loc_est_max = max(loc_est_max, x_j)
                self.x_i = loc_est_max

                msg = MsgFloat()
                msg.data = [float(self.agent_id), float(self.k), float(loc_est_max)]
                self.pub.publish(msg)
                print(f"Agent {self.agent_id} at time {self.k} has value {loc_est_max}")
                self.k += 1

                if self.k > 10:
                    print("Maximum iteration reached. Shutting down!")
                    self.destroy_node()
                

def main():
    rclpy.init()
    anAgent = TheAgent()
    sleep(1)
    rclpy.spin(anAgent)
    anAgent.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()