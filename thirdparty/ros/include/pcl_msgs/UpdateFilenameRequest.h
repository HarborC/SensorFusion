// Generated by gencpp from file pcl_msgs/UpdateFilenameRequest.msg
// DO NOT EDIT!


#ifndef PCL_MSGS_MESSAGE_UPDATEFILENAMEREQUEST_H
#define PCL_MSGS_MESSAGE_UPDATEFILENAMEREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace pcl_msgs
{
template <class ContainerAllocator>
struct UpdateFilenameRequest_
{
  typedef UpdateFilenameRequest_<ContainerAllocator> Type;

  UpdateFilenameRequest_()
    : filename()  {
    }
  UpdateFilenameRequest_(const ContainerAllocator& _alloc)
    : filename(_alloc)  {
  (void)_alloc;
    }



   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _filename_type;
  _filename_type filename;





  typedef boost::shared_ptr< ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator> const> ConstPtr;

}; // struct UpdateFilenameRequest_

typedef ::pcl_msgs::UpdateFilenameRequest_<std::allocator<void> > UpdateFilenameRequest;

typedef boost::shared_ptr< ::pcl_msgs::UpdateFilenameRequest > UpdateFilenameRequestPtr;
typedef boost::shared_ptr< ::pcl_msgs::UpdateFilenameRequest const> UpdateFilenameRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator1> & lhs, const ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator2> & rhs)
{
  return lhs.filename == rhs.filename;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator1> & lhs, const ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace pcl_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "030824f52a0628ead956fb9d67e66ae9";
  }

  static const char* value(const ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x030824f52a0628eaULL;
  static const uint64_t static_value2 = 0xd956fb9d67e66ae9ULL;
};

template<class ContainerAllocator>
struct DataType< ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "pcl_msgs/UpdateFilenameRequest";
  }

  static const char* value(const ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "string filename\n"
;
  }

  static const char* value(const ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.filename);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct UpdateFilenameRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::pcl_msgs::UpdateFilenameRequest_<ContainerAllocator>& v)
  {
    s << indent << "filename: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.filename);
  }
};

} // namespace message_operations
} // namespace ros

#endif // PCL_MSGS_MESSAGE_UPDATEFILENAMEREQUEST_H
