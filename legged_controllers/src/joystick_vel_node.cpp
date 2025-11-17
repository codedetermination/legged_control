// joystick_vel_node.cpp
#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <geometry_msgs/Twist.h>

class JoyVel {
public:
    JoyVel() {
        ros::NodeHandle nh;
        ros::NodeHandle pnh("~");

        pnh.param("axis_vx", axis_vx_, 1);      // 左摇杆上下
        pnh.param("axis_vy", axis_vy_, 0);      // 左摇杆左右
        pnh.param("axis_wz", axis_wheight_, 3);      // 右摇杆上下
        pnh.param("axis_wz", axis_wz_, 2);      // 右摇杆上下
        pnh.param("max_vx",  max_vx_,  1.0);
        pnh.param("max_vy",  max_vy_,  0.5);
        pnh.param("max_wz",  max_wz_,  1.0);

        joy_sub_ = nh.subscribe("/joy", 1, &JoyVel::joyCallback, this);
        cmd_pub_ = nh.advertise<geometry_msgs::Twist>("target_vels", 1);
    }

    void joyCallback(const sensor_msgs::Joy::ConstPtr& msg) {
        geometry_msgs::Twist cmd;
        cmd.linear.x  = msg->axes[axis_vx_] * max_vx_;
        cmd.linear.y  = msg->axes[axis_vy_] * max_vy_;
        cmd.linear.z  = msg->axes[axis_wheight_] * max_vx_;
        cmd.angular.z = msg->axes[axis_wz_] * max_wz_;
        cmd_pub_.publish(cmd);
    }

private:
    ros::Subscriber joy_sub_;
    ros::Publisher  cmd_pub_;
    int axis_vx_, axis_vy_, axis_wz_,axis_wheight_;
    double max_vx_, max_vy_, max_wz_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "joy_vel");
    JoyVel node;
    ros::spin();
    return 0;
}
