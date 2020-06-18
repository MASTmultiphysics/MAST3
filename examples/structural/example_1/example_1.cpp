#include <mast/base/exceptions.h>

int main() {
 
    int a = 1, b = 0;
    //Assert1(1 == 0, 1, "Checking stuff");
    Assert2(a == b, a, b, "Checking stuff");
}
