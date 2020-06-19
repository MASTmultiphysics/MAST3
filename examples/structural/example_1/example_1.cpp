#include <mast/base/exceptions.hpp>
#include <mast/util/perf_log.hpp>

int main() {
 
    int a = 1, b = 0;
    //Assert1(1 == 0, 1, "Checking stuff");
    //Assert2(a == b, a, b, "Checking stuff");
    Error(false, "This should quit");
}
