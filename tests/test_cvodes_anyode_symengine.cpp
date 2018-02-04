// C++11 source code.
#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
#include "catch.hpp"
#include "cvodes_anyode.hpp"
#include "testing_utils.hpp"

class SymbolicSys {
    RCP<const Basic> m_t;
    DenseMatrix m_Y;
    vec_basic m_p;
    DenseMatrix m_E, m_J;
    LLVMDoubleVisitor m_rhs_vis;
    LLVMDoubleVisitor m_jac_vis;
    SymbolicSys(RCP<const Basic> t, vec_basic y, vec_basic p, vec_basic exprs) :
        m_t(t), m_Y(DenseMatrix(y.size(), 1, y)), m_p(p), m_E(DenseMatrix(exprs.size(), 1, exprs))
    {
        jacobian(m_E, m_Y, m_J);
        m_rhs_vis.init(..., m_J.transpose().flat());
    }

};

TEST_CASE( "decay_adaptive", "[simple_adaptive]" ) {
    Decay<double> odesys(1.0);
    std::vector<int> root_indices;
    int td = 1;
    double * xyout = (double *)malloc(td*(odesys.get_ny()+1)*sizeof(double));
#define xout(ti) xyout[2*ti]
#define yout(ti) xyout[2*ti + 1]
    xout(0) = 0.0;
    yout(0) = 1.0;
    auto nout = cvodes_anyode::simple_adaptive(&xyout, &td, &odesys, {1e-10}, 1e-10, cvodes_cxx::LMM::Adams, 1.0, root_indices);
    for (int i = 0; i < nout; ++i){
        REQUIRE( std::abs(std::exp(-xout(i)) - yout(i)) < 1e-8 );
    }
    REQUIRE( odesys.last_integration_info["n_steps"] > 1 );
    REQUIRE( odesys.last_integration_info["n_steps"] >= nout );
    REQUIRE( odesys.last_integration_info["n_steps"] < 997 );
#undef xout
#undef yout
    free(xyout);
}
