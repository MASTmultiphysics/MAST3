
#ifndef __mast__perflog_h__
#define __mast__perflog_h__

#include <sys/time.h>
#include <sys/resource.h>
#include <map>
#include <stack>

#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

namespace MAST {

namespace Utility {

class PerformanceLogging {
  
public:

    struct EventData {
      
        EventData(): sys(0.), user(0.), ncalls(0) { }
        
        Real       sys;
        Real       user;
        uint_type  ncalls;
    };
    
    
    class Event {
      
    public:
        
        inline Event(const std::string& nm):
        _name_ref   (nm),
        _sys_time   (0.),
        _user_time  (0.)
        { }

        
        inline Event(const MAST::Utility::PerformanceLogging::Event& e):
        _name_ref   (e._name_ref),
        _sys_time   (e._sys_time),
        _user_time  (e._user_time)
        { }

        inline ~Event() { }
        
        inline const std::string& name() const { return _name_ref;}
        
        inline void start() {
            
            rusage r;
            getrusage(RUSAGE_SELF, &r);
            _sys_time   = - (r.ru_stime.tv_sec + 1.e-6 * r.ru_stime.tv_usec);
            _user_time  = - (r.ru_utime.tv_sec + 1.e-6 * r.ru_utime.tv_usec);
        }
        
        inline void stop() {
            
            rusage r;
            getrusage(RUSAGE_SELF, &r);
            _sys_time  += r.ru_stime.tv_sec + 1.e-6 * r.ru_stime.tv_usec;
            _user_time += r.ru_utime.tv_sec + 1.e-6 * r.ru_utime.tv_usec;
        }
        
        inline Real get_sys_time  () const { return  _sys_time; }
        inline Real get_user_time () const { return _user_time; }

    private:
        
        const std::string& _name_ref;
        Real               _sys_time;
        Real               _user_time;
    };
    
    using map_type   = std::map<std::string, MAST::Utility::PerformanceLogging::EventData>;
    using stack_type = std::stack<MAST::Utility::PerformanceLogging::Event>;
    
    PerformanceLogging();
    
    ~PerformanceLogging();
    
    inline void start_event(const std::string& nm) {
        
        map_type::iterator it = _get_event_data(nm);
        _event_stack.push(MAST::Utility::PerformanceLogging::Event(it->first));
    }

    inline void stop_event(const std::string& nm) {
        

        // the object cannot be empty
        Assert1(!_event_stack.empty(), !_event_stack.empty(),
                "Empty event stack");
        
        const MAST::Utility::PerformanceLogging::Event&
        v = _event_stack.top();
        
        // make sure that the top object has the same name as provided here
        Assert0(nm == v.name(),
                "Name provided must match as the last started event.");

        map_type::iterator it = _get_event_data(nm);
        it->second.ncalls++;
        it->second.sys  += v.get_sys_time();
        it->second.user += v.get_user_time();
        
        _event_stack.pop();
    }

    
protected:
    
    inline map_type::iterator _get_event_data(const std::string& nm) {
        
        map_type::iterator
        it   = _event_time.find(nm),
        end  = _event_time.end();
        
        if (it == end)
            it = _event_time.insert(std::make_pair(nm, MAST::Utility::PerformanceLogging::EventData())).first;

        return it;
    }

    stack_type _event_stack;
    map_type   _event_time;
};
}
}

#endif // __mast__perflog_h__

