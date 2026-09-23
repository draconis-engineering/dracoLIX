#include "../../core/include/dracolix/dracolix.hpp"
#include "../../core/include/dracolix/numerics.hpp"
#include "../../core/include/dracolix/ode.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace dracolix;

void test_roots(){
    double r = numerics::bisect([](double x){return x*x-2;}, 0, 2);
    assert(std::abs(r - std::sqrt(2))<1e-7);
    double rn = numerics::newton([](double x){return x*x-2;}, [](double x){return 2*x;}, 1.5);
    assert(std::abs(rn - std::sqrt(2))<1e-8);
    std::cout<<" roots ok\n";
}
void test_interp_diff_integ(){
    std::vector<double> xs={0,1,2}, ys={0,1,4};
    assert(std::abs(numerics::interp_linear(xs,ys,1.5)-2.5)<1e-9);
    double d = numerics::derivative([](double x){return x*x;}, 2.0);
    assert(std::abs(d-4.0)<1e-4);
    double I = numerics::integrate_simpson([](double x){return x*x;}, 0,1, 100);
    assert(std::abs(I-1.0/3)<1e-6);
    double It = numerics::integrate_trapezoidal([](double x){return x*x;},0,1, 1000);
    assert(std::abs(It-1.0/3)<1e-3);
    std::cout<<" interp/diff/integ ok\n";
}
void test_ode(){
    // dy/dt = y, y0=1 => y=exp(t), t=1 => e
    auto f = [](double, const ode::State& y){ return ode::State{y[0]}; };
    auto traj = ode::rk4(f, 0, {1.0}, 1.0, 0.1);
    assert(std::abs(traj.back()[0] - std::exp(1.0))<1e-3);
    auto traj2 = ode::euler(f, 0, {1.0}, 1.0, 0.01);
    assert(std::abs(traj2.back()[0] - std::exp(1.0))<0.05);
    auto traj3 = ode::adaptive_rk45(f, 0, {1.0}, 1.0, 1e-8, 0.2);
    assert(std::abs(traj3.back()[0] - std::exp(1.0))<1e-5);
    // harmonic: y''=-y => state [y, v]
    auto f2 = [](double, const ode::State& y){ return ode::State{y[1], -y[0]}; };
    auto th = ode::rk4(f2, 0, {1.0,0.0}, 6.283185307, 0.1);
    assert(std::abs(th.back()[0]-1.0)<1e-2);
    std::cout<<" ode ok\n";
}
int main(){ test_roots(); test_interp_diff_integ(); test_ode(); std::cout<<"All numerics/ode tests passed\n"; return 0; }
