/*
* MAST: Multidisciplinary-design Adaptation and Sensitivity Toolkit
* Copyright (C) 2013-2020  Manav Bhatia and MAST authors
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with this library; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

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
        
        real_t      sys;
        real_t      user;
        uint_t      ncalls;
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
        
        inline real_t get_sys_time  () const { return  _sys_time; }
        inline real_t get_user_time () const { return _user_time; }

    private:
        
        const std::string& _name_ref;
        real_t              _sys_time;
        real_t              _user_time;
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

}  // namespace Utility
}  // namespace MAST

#endif // __mast__perflog_h__

