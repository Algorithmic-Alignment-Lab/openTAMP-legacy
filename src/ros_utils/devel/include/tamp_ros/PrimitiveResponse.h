// Generated by gencpp from file tamp_ros/PrimitiveResponse.msg
// DO NOT EDIT!


#ifndef TAMP_ROS_MESSAGE_PRIMITIVERESPONSE_H
#define TAMP_ROS_MESSAGE_PRIMITIVERESPONSE_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace tamp_ros
{
template <class ContainerAllocator>
struct PrimitiveResponse_
{
  typedef PrimitiveResponse_<ContainerAllocator> Type;

  PrimitiveResponse_()
    : task_distr()
    , obj_distr()
    , targ_distr()  {
    }
  PrimitiveResponse_(const ContainerAllocator& _alloc)
    : task_distr(_alloc)
    , obj_distr(_alloc)
    , targ_distr(_alloc)  {
  (void)_alloc;
    }



   typedef std::vector<float, typename ContainerAllocator::template rebind<float>::other >  _task_distr_type;
  _task_distr_type task_distr;

   typedef std::vector<float, typename ContainerAllocator::template rebind<float>::other >  _obj_distr_type;
  _obj_distr_type obj_distr;

   typedef std::vector<float, typename ContainerAllocator::template rebind<float>::other >  _targ_distr_type;
  _targ_distr_type targ_distr;





  typedef boost::shared_ptr< ::tamp_ros::PrimitiveResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::tamp_ros::PrimitiveResponse_<ContainerAllocator> const> ConstPtr;

}; // struct PrimitiveResponse_

typedef ::tamp_ros::PrimitiveResponse_<std::allocator<void> > PrimitiveResponse;

typedef boost::shared_ptr< ::tamp_ros::PrimitiveResponse > PrimitiveResponsePtr;
typedef boost::shared_ptr< ::tamp_ros::PrimitiveResponse const> PrimitiveResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::tamp_ros::PrimitiveResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::tamp_ros::PrimitiveResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace tamp_ros

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'tamp_ros': ['/home/michaelmcdonald/dependencies/tampy/src/ros_utils/src/tamp_ros/msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::tamp_ros::PrimitiveResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::tamp_ros::PrimitiveResponse_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::tamp_ros::PrimitiveResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::tamp_ros::PrimitiveResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::tamp_ros::PrimitiveResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::tamp_ros::PrimitiveResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::tamp_ros::PrimitiveResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "e7c5b74c5540db8867d8b5db7fc110e8";
  }

  static const char* value(const ::tamp_ros::PrimitiveResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xe7c5b74c5540db88ULL;
  static const uint64_t static_value2 = 0x67d8b5db7fc110e8ULL;
};

template<class ContainerAllocator>
struct DataType< ::tamp_ros::PrimitiveResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "tamp_ros/PrimitiveResponse";
  }

  static const char* value(const ::tamp_ros::PrimitiveResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::tamp_ros::PrimitiveResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "\n\
float32[] task_distr\n\
float32[] obj_distr\n\
float32[] targ_distr\n\
\n\
";
  }

  static const char* value(const ::tamp_ros::PrimitiveResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::tamp_ros::PrimitiveResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.task_distr);
      stream.next(m.obj_distr);
      stream.next(m.targ_distr);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct PrimitiveResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::tamp_ros::PrimitiveResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::tamp_ros::PrimitiveResponse_<ContainerAllocator>& v)
  {
    s << indent << "task_distr[]" << std::endl;
    for (size_t i = 0; i < v.task_distr.size(); ++i)
    {
      s << indent << "  task_distr[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.task_distr[i]);
    }
    s << indent << "obj_distr[]" << std::endl;
    for (size_t i = 0; i < v.obj_distr.size(); ++i)
    {
      s << indent << "  obj_distr[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.obj_distr[i]);
    }
    s << indent << "targ_distr[]" << std::endl;
    for (size_t i = 0; i < v.targ_distr.size(); ++i)
    {
      s << indent << "  targ_distr[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.targ_distr[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // TAMP_ROS_MESSAGE_PRIMITIVERESPONSE_H
