// Generated by gencpp from file tamp_ros/Primitive.msg
// DO NOT EDIT!


#ifndef TAMP_ROS_MESSAGE_PRIMITIVE_H
#define TAMP_ROS_MESSAGE_PRIMITIVE_H

#include <ros/service_traits.h>


#include <tamp_ros/PrimitiveRequest.h>
#include <tamp_ros/PrimitiveResponse.h>


namespace tamp_ros
{

struct Primitive
{

typedef PrimitiveRequest Request;
typedef PrimitiveResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct Primitive
} // namespace tamp_ros


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::tamp_ros::Primitive > {
  static const char* value()
  {
    return "ec8948c09b640bcf5ec37fe64f2d51b1";
  }

  static const char* value(const ::tamp_ros::Primitive&) { return value(); }
};

template<>
struct DataType< ::tamp_ros::Primitive > {
  static const char* value()
  {
    return "tamp_ros/Primitive";
  }

  static const char* value(const ::tamp_ros::Primitive&) { return value(); }
};


// service_traits::MD5Sum< ::tamp_ros::PrimitiveRequest> should match 
// service_traits::MD5Sum< ::tamp_ros::Primitive > 
template<>
struct MD5Sum< ::tamp_ros::PrimitiveRequest>
{
  static const char* value()
  {
    return MD5Sum< ::tamp_ros::Primitive >::value();
  }
  static const char* value(const ::tamp_ros::PrimitiveRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::tamp_ros::PrimitiveRequest> should match 
// service_traits::DataType< ::tamp_ros::Primitive > 
template<>
struct DataType< ::tamp_ros::PrimitiveRequest>
{
  static const char* value()
  {
    return DataType< ::tamp_ros::Primitive >::value();
  }
  static const char* value(const ::tamp_ros::PrimitiveRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::tamp_ros::PrimitiveResponse> should match 
// service_traits::MD5Sum< ::tamp_ros::Primitive > 
template<>
struct MD5Sum< ::tamp_ros::PrimitiveResponse>
{
  static const char* value()
  {
    return MD5Sum< ::tamp_ros::Primitive >::value();
  }
  static const char* value(const ::tamp_ros::PrimitiveResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::tamp_ros::PrimitiveResponse> should match 
// service_traits::DataType< ::tamp_ros::Primitive > 
template<>
struct DataType< ::tamp_ros::PrimitiveResponse>
{
  static const char* value()
  {
    return DataType< ::tamp_ros::Primitive >::value();
  }
  static const char* value(const ::tamp_ros::PrimitiveResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // TAMP_ROS_MESSAGE_PRIMITIVE_H
