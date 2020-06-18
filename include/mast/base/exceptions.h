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

#ifndef __mast__exceptions__
#define __mast__exceptions__

#include <iostream>
#include <string>

namespace MAST {

namespace Exceptions {

class ExceptionBase {

public:
    ExceptionBase(const std::string& cond,
                  const std::string& msg,
                  const std::string& file,
                  const int line):
    _cond  (cond),
    _msg   (msg),
    _file  (file),
    _line  (line)
    { }

    virtual ~ExceptionBase() { }
    
    void throw_error() {
        
        std::cerr
        << "Condition Violated: "
        << _cond << std::endl
        << _error_message() << std::endl
        << _file << " : " << _line << std::endl;
        
        throw;
    }
    
protected:
    
    virtual const std::string& _error_message() = 0;
        
    const std::string _cond;
    const std::string _msg;
    const std::string _file;
    const int         _line;
};




template <typename Val1Type>
class Exception1:
public MAST::Exceptions::ExceptionBase {
  
public:
    
    Exception1(const Val1Type& v1,
               const std::string& cond,
               const std::string& msg,
               const std::string& file,
               const int          line):
    MAST::Exceptions::ExceptionBase(cond, msg, file, line),
    _val1    (v1),
    _valstr  ("Val 1: " + std::to_string(v1))
    {}

    virtual ~Exception1() { }
    
protected:
    
    virtual const std::string& _error_message() override { return _valstr;}
    
    const Val1Type     _val1;
    const std::string  _valstr;
};


template <typename Val1Type, typename Val2Type>
class Exception2:
public MAST::Exceptions::ExceptionBase {
  
public:
    
    Exception2(const Val1Type& v1,
               const Val2Type& v2,
               const std::string& cond,
               const std::string& msg,
               const std::string& file,
               const int          line):
    MAST::Exceptions::ExceptionBase(cond,  msg, file, line),
    _val1    (v1),
    _val2    (v2),
    _valstr  ("Val 1: " + std::to_string(v1) + " \nVal 2: " + std::to_string(v2))
    { }
    
protected:
    
    virtual const std::string& _error_message() override { return _valstr;}

    Val1Type  _val1;
    Val2Type  _val2;
    const std::string  _valstr;
};
}
}


#ifdef NDEBUG

#define Assert1(cond, v1, msg) { }
#define Assert2(cond, v1, v2, msg)  { }

#else

#define Assert1(cond, v1, msg)                                 \
    if (!(cond))                                               \
    MAST::Exceptions::Exception1<decltype(v1)>                 \
                                (v1,                           \
                                 #cond,                        \
                                 msg,                          \
                                 __FILE__,                     \
                                 __LINE__).throw_error()       \
                                 
#define Assert2(cond, v1, v2,  msg)                            \
    if (!(cond))                                               \
    MAST::Exceptions::Exception2<decltype(v1), decltype(v2)>   \
                                (v1,                           \
                                 v2,                           \
                                 #cond,                        \
                                 msg,                          \
                                 __FILE__,                     \
                                 __LINE__).throw_error()       \

#endif // DEBUG

#endif // __mast__exceptions__
