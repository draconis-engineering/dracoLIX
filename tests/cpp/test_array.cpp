#include "../../core/include/dracolix/array.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

using namespace dracolix;

void test_layout() {
    Array<double> r({2,3}, Layout::RowMajor);
    Array<double> c({2,3}, Layout::ColMajor);
    assert(r.strides()[0]==3 && r.strides()[1]==1);
    assert(c.strides()[0]==1 && c.strides()[1]==2);
    std::cout << " layout OK\n";
}

void test_view_transpose() {
    Array<double> a({2,3});
    // fill 0..5 row-major
    for(size_t i=0;i<6;++i) a[i]=(double)i;
    // a = [[0 1 2],[3 4 5]]
    auto vt = a.transpose_view();
    assert(vt.shape()[0]==3 && vt.shape()[1]==2);
    assert(vt.at(0,1)==3);
    assert(vt.at(2,0)==2);
    assert(vt.at(2,1)==5);
    // mutate via view should reflect
    vt.at(0,0)=99;
    assert(a.at(0,0)==99);
    a.at(0,0)=0; // restore

    auto mt = a.transpose();
    assert(mt.shape()[0]==3 && mt.shape()[1]==2);
    assert(mt.at(0,1)==3);
    mt.at(0,0)=42;
    assert(a.at(0,0)==0); // copy not affected
    std::cout << " transpose OK\n";
}

void test_permute() {
    Array<double> a({2,3,4});
    for(size_t i=0;i<a.size();++i) a[i]=(double)i;
    auto v = a.permute({2,0,1}); // shape 4,2,3
    assert(v.shape()[0]==4 && v.shape()[1]==2 && v.shape()[2]==3);
    // check mapping: original (1,2,3) -> permuted (3,1,2)
    assert(v.at(std::vector<size_t>{3,1,2}) == a.at(std::vector<size_t>{1,2,3}));
    std::cout << " permute OK\n";
}

void test_slice() {
    Array<double> a({6});
    for(size_t i=0;i<6;++i) a[i]=(double)i;
    auto s = a.slice(Slice::range(1,5,2)); // 1,3
    assert(s.shape()[0]==2);
    assert(s.at(std::vector<size_t>{0})==1);
    assert(s.at(std::vector<size_t>{1})==3);
    s.at(std::vector<size_t>{0})=99;
    assert(a[1]==99);
    a[1]=1;
    // 2-D slice
    Array<double> b({3,4});
    for(size_t i=0;i<12;++i) b[i]=i;
    auto v = b.slice({Slice::range(0,2), Slice::all()});
    assert(v.shape()[0]==2 && v.shape()[1]==4);
    assert(v.at(1,3)==7);
    // negative step
    auto rev = a.slice(Slice::range(5, -1, -1));
    assert(rev.shape()[0]==6);
    assert(rev.at(std::vector<size_t>{0})==5);
    assert(rev.at(std::vector<size_t>{5})==0);
    std::cout << " slice OK\n";
}

void test_reductions() {
    Array<double> a({2,3});
    // [[1 2 3],[4 5 6]]
    a.at(0,0)=1; a.at(0,1)=2; a.at(0,2)=3;
    a.at(1,0)=4; a.at(1,1)=5; a.at(1,2)=6;
    assert(a.sum()==21);
    assert(a.min()==1);
    assert(a.max()==6);
    assert(std::abs(a.mean()-3.5)<1e-9);
    // axis 0 -> sum over rows => [5,7,9]
    auto s0 = a.sum(0);
    assert(s0.shape()[0]==3);
    assert(s0[0]==5 && s0[1]==7 && s0[2]==9);
    auto s1 = a.sum(1);
    assert(s1.shape()[0]==2);
    assert(s1[0]==6 && s1[1]==15);
    assert(a.min_axis(0)[0]==1);
    assert(a.max_axis(1)[1]==6);
    assert(std::abs(a.mean_axis(0)[0]-2.5)<1e-9);
    std::cout << " reductions OK\n";
}

void test_astype() {
    Array<double> a({3});
    a[0]=1.7; a[1]=2.2; a[2]=3.9;
    auto b = a.astype<int32_t>();
    assert(b[0]==1 && b[2]==3);
    assert(b.dtype()==DType::I32);
    auto c = b.astype<double>();
    assert(c[0]==1.0);
    std::cout << " astype OK\n";
}

void test_broadcast() {
    Array<double> a({2,1}); a[0]=1; a[1]=2;
    Array<double> b({1,3}); b[0]=10; b[1]=20; b[2]=30;
    auto c = a + b;
    assert((c.shape()==std::vector<size_t>{2,3}));
    assert(c.at(0,0)==11 && c.at(0,2)==31 && c.at(1,2)==32);
    Array<double> row({3}); row[0]=100; row[1]=200; row[2]=300;
    Array<double> m({2,3}); for(size_t i=0;i<6;++i) m[i]=(double)i;
    auto d = m + row;
    assert(d.at(0,0)==100 && d.at(1,2)==305);
    // 3D
    Array<double> e({2,3,1}); for(size_t i=0;i<6;++i) e[i]=(double)i;
    Array<double> f({3}); f[0]=10; f[1]=20; f[2]=30;
    auto g = e + f;
    assert((g.shape()==std::vector<size_t>{2,3,3}));
    try { Array<double> h({2,3}); Array<double> k({3,2}); auto l=h+k; assert(false); }
    catch(const std::invalid_argument&){ }
    assert((a * b).at(1,2)==60);
    std::cout << " broadcast OK\n";
}

int main() {
    test_layout();
    test_view_transpose();
    test_permute();
    test_slice();
    test_reductions();
    test_astype();
    test_broadcast();
    // existing smoke
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
    std::cout << "All Phase1 tests passed\n";
    return 0;
}
