// Generated by gencpp from file tamp_ros/PolicyProb.msg
// DO NOT EDIT!


#ifndef TAMP_ROS_MESSAGE_POLICYPROB_H
#define TAMP_ROS_MESSAGE_POLICYPROB_H

#include <ros/service_traits.h>


#include <tamp_ros/PolicyProbRequest.h>
#include <tamp_ros/PolicyProbResponse.h>


namespace tamp_ros
{

struct PolicyProb
{

typedef PolicyProbRequest Request;
typedef PolicyProbResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct PolicyProb
} // namespace tamp_ros


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::tamp_ros::PolicyProb > {
  static const char* value()
  {
    return "543016ad28d3afef79460f66829d896a";
  }

  static const char* value(const ::tamp_ros::PolicyProb&) { return value(); }
};

template<>
struct DataType< ::tamp_ros::PolicyProb > {
  static const char* value()
  {
    return "tamp_ros/PolicyProb";
  }

  static const char* value(const ::tamp_ros::PolicyProb&) { return value(); }
};


// service_traits::MD5Sum< ::tamp_ros::PolicyProbRequest> should match 
// service_traits::MD5Sum< ::tamp_ros::PolicyProb > 
template<>
struct MD5Sum< ::tamp_ros::PolicyProbRequest>
{
  static const char* value()
  {
    return MD5Sum< ::tamp_ros::PolicyProb >::value();
  }
  static const char* value(const ::tamp_ros::PolicyProbRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::tamp_ros::PolicyProbRequest> should match 
// service_traits::DataType< ::tamp_ros::PolicyProb > 
template<>
struct DataType< ::tamp_ros::PolicyProbRequest>
{
  static const char* value()
  {
    return DataType< ::tamp_ros::PolicyProb >::value();
  }
  static const char* value(const ::tamp_ros::PolicyProbRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::tamp_ros::PolicyProbResponse> should match 
// service_traits::MD5Sum< ::tamp_ros::PolicyProb > 
template<>
struct MD5Sum< ::tamp_ros::PolicyProbResponse>
{
  static const char* value()
  {
    return MD5Sum< ::tamp_ros::PolicyProb >::value();
  }
  static const char* value(const ::tamp_ros::PolicyProbResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::tamp_ros::PolicyProbResponse> should match 
// service_traits::DataType< ::tamp_ros::PolicyProb > 
template<>
struct DataType< ::tamp_ros::PolicyProbResponse>
{
  static const char* value()
  {
    return DataType< ::tamp_ros::PolicyProb >::value();
  }
  static const char* value(const ::tamp_ros::PolicyProbResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // TAMP_ROS_MESSAGE_POLICYPROB_H