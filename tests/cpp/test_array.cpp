
#include "../../core/include/dracolix/array.hpp"
#include <cassert>
#include <iostream>


int main() {
    using namespace dracolix;
    Array<double> a({2,3});
    assert(a.size()==6 && a.ndim()==2);
    a.at(0,0)=1; a.at(1,2)=6;
    assert(a.at(0,0)==1 && a.at(1,2)==6);
    Array<double> b({2,3});
    b = b + 1.0;
    auto c = a + b;
    (void)c;
    a.reshape({6});
    assert(a.ndim()==1 && a.size()==6);
    std::cout << "Array smoke test passed\n";
    return 0;
}
