#include <cinolib/gl/glcanvas.h>
#include <cinolib/gl/surface_mesh_controls.h>
#include <cinolib/matrix_eigenfunctions.h>

using namespace cinolib;

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

Eigen::SparseMatrix<double> Laplacian_matrix(const AbstractPolygonMesh<> & m)
{
    std::vector<Eigen::Triplet<double>> entries;
    for(uint vid=0; vid<m.num_verts(); ++vid)
    {
        for(uint nbr : m.adj_v2v(vid))
        {
            entries.emplace_back(vid,nbr,1.0);
        }
        entries.emplace_back(vid,vid,-double(m.vert_valence(vid)));
    }
    Eigen::SparseMatrix<double> L(m.num_verts(),m.num_verts());
    L.setFromTriplets(entries.begin(), entries.end());
    return L;
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

int main(int argc, char **argv)
{
    DrawableTrimesh<> m(argv[1]);
    m.show_vert_color();


    Eigen::SparseMatrix<double> L = Laplacian_matrix(m);
    uint n_eigs = std::min(300, int(m.num_verts()/3));
    std::vector<double> eigs, e_min, e_max;
    matrix_eigenfunctions(L, true, n_eigs, eigs, e_min, e_max);

    // pack eigenfunctions to a dense Eigen matrix
    Eigen::MatrixXd E(m.num_verts(),n_eigs);
    for(uint e=0; e<n_eigs; ++e)
    {
        for(uint vid=0; vid<m.num_verts(); ++vid)
        {
            E.coeffRef(vid,e) = eigs.at(e*m.num_verts()+vid);
        }
    }

    // pack X,Y,Z Euclidean signals
    Eigen::MatrixXd X(m.num_verts(),3);
    for(uint vid=0; vid<m.num_verts(); ++vid)
    {
        X.coeffRef(vid,0) = m.vert(vid).x();
        X.coeffRef(vid,1) = m.vert(vid).y();
        X.coeffRef(vid,2) = m.vert(vid).z();
    }

    Eigen::MatrixXd X_coeff = E.transpose()*X; // spectral coefficients
    Eigen::MatrixXd X_recon = E * X_coeff;     // reconstruction
    for(uint vid=0; vid<m.num_verts(); ++vid)
    {
        m.vert(vid) = vec3d(X_recon.coeff(vid,0),
                            X_recon.coeff(vid,1),
                            X_recon.coeff(vid,2));
    }
    m.updateGL();

    GLcanvas gui;
    gui.push(&m);
    gui.push(new SurfaceMeshControls<DrawableTrimesh<>>(&m,&gui));

    int n_funcs = n_eigs;
    int f = 1;
    gui.callback_app_controls = [&]()
    {
        if(ImGui::SliderInt("Eigenfunctions", &f, 1, n_eigs-1))
        {
            for(uint vid=0; vid<m.num_verts(); ++vid)
            {
                double val  = eigs.at(f*m.num_verts()+vid);
                double norm = (val-e_min[f])/(e_max[f]-e_min[f]);
                m.vert_data(vid).color = Color::red_white_blue_ramp_01(norm);
            }
            m.show_vert_color();
        }

        if(ImGui::SliderInt("Reconstruction", &n_funcs, 1, n_eigs-1))
        {
            for(uint vid=0; vid<m.num_verts(); ++vid)
            {
                m.vert(vid) = vec3d(0,0,0);
                for(int i=0; i<n_funcs; ++i)
                {
                    m.vert(vid).x() += E.coeff(vid,i) * X_coeff.coeff(i,0);
                    m.vert(vid).y() += E.coeff(vid,i) * X_coeff.coeff(i,1);
                    m.vert(vid).z() += E.coeff(vid,i) * X_coeff.coeff(i,2);
                }
            }
            m.updateGL();
        }                
    };

    return gui.launch();
}
