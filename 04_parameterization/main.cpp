#include <cinolib/gl/glcanvas.h>
#include <cinolib/gl/surface_mesh_controls.h>
#include <cinolib/meshes/meshes.h>
#include <cinolib/geometry/n_sided_poygon.h>
#include <cinolib/linear_solvers.h>
#include <cinolib/profiler.h>

using namespace cinolib;

Profiler p;

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

void laplacian(DrawableTrimesh<> & m, const uint vid)
{
    assert(!m.vert_is_boundary(vid));
    vec3d p(0,0,0);
    for(uint nbr : m.adj_v2v(vid)) p += m.vert(nbr);
    m.vert(vid) = p/m.vert_valence(vid);
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

void laplacian(DrawableTrimesh<> & m, const int n_iters)
{
    for(int i=0; i<n_iters; ++i)
    {
        for(uint vid=0; vid<m.num_verts(); ++vid)
        {
            if(m.vert_is_boundary(vid)) continue;
            laplacian(m,vid);
        }
    }
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

Eigen::SparseMatrix<double> laplacian_matrix3(const DrawableTrimesh<> & m)
{
    std::vector<Eigen::Triplet<double>> entries;
    uint nv = m.num_verts();
    for(uint vid=0; vid<nv; ++vid)
    {
        entries.emplace_back(     vid,     vid,double(m.vert_valence(vid)));
        entries.emplace_back(  nv+vid,  nv+vid,double(m.vert_valence(vid)));
        entries.emplace_back(2*nv+vid,2*nv+vid,double(m.vert_valence(vid)));

        for(uint nbr : m.adj_v2v(vid))
        {
            entries.emplace_back(     vid,     nbr,-1);
            entries.emplace_back(  nv+vid,  nv+nbr,-1);
            entries.emplace_back(2*nv+vid,2*nv+nbr,-1);
        }
    }
    Eigen::SparseMatrix<double> L(m.num_verts()*3,m.num_verts()*3);
    L.setFromTriplets(entries.begin(),entries.end());
    return L;
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

Eigen::SparseMatrix<double> laplacian_matrix(const DrawableTrimesh<> & m)
{
    std::vector<Eigen::Triplet<double>> entries;
    uint nv = m.num_verts();
    for(uint vid=0; vid<nv; ++vid)
    {
        entries.emplace_back(vid,vid,double(m.vert_valence(vid)));

        for(uint nbr : m.adj_v2v(vid))
        {
            entries.emplace_back(vid,nbr,-1);
        }
    }
    Eigen::SparseMatrix<double> L(m.num_verts(),m.num_verts());
    L.setFromTriplets(entries.begin(),entries.end());
    return L;
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

void solve(const Eigen::SparseMatrix<double> & L,
           const Eigen::VectorXd             & rhs_x,
           const Eigen::VectorXd             & rhs_y,
           const std::map<uint,double>       & bc_x,
           const std::map<uint,double>       & bc_y,
                 Eigen::VectorXd             & x,
                 Eigen::VectorXd             & y)
{
    std::vector<int> col_map(L.rows(), 0);
    for(const auto & obj : bc_x)
    {
        col_map[obj.first] = -1;
    }
    uint fresh_id = 0;
    for(uint col=0; col<L.cols(); ++col)
    {
        if(col_map.at(col)==0)
        {
            col_map[col] = fresh_id++;
        }
    }

    uint size = uint(L.rows() - bc_x.size());
    Eigen::VectorXd rhs_x_prime = Eigen::VectorXd::Zero(size);
    Eigen::VectorXd rhs_y_prime = Eigen::VectorXd::Zero(size);

    std::vector<Entry> entries;

    // iterate over the non-zero entries of sparse matrix A
    for(uint i=0; i<L.outerSize(); ++i)
    {
        for(Eigen::SparseMatrix<double>::InnerIterator it(L,i); it; ++it)
        {
            uint   row = uint(it.row());
            uint   col = uint(it.col());
            double val = it.value();

            // Laplacian of a constrained vertex
            if(col_map[row]<0) continue;

            if(col_map[col]<0)
            {
                rhs_x_prime[col_map[row]] -= bc_x.at(col) * val;
                rhs_y_prime[col_map[row]] -= bc_y.at(col) * val;
            }
            else
            {
                entries.emplace_back(col_map[row],col_map[col],val);
            }
        }
    }

    Eigen::SparseMatrix<double> Aprime(size, size);
    Aprime.setFromTriplets(entries.begin(), entries.end());

    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver(Aprime);
    assert(solver.info() == Eigen::Success);
    Eigen::VectorXd tmp_x = solver.solve(rhs_x_prime).eval();
    Eigen::VectorXd tmp_y = solver.solve(rhs_y_prime).eval();

    x.resize(L.cols());
    y.resize(L.cols());
    for(uint col=0; col<L.cols(); ++col)
    {
        if(col_map[col]>=0)
        {
            x[col] = tmp_x[col_map[col]];
            y[col] = tmp_y[col_map[col]];
        }
        else
        {
            x[col] = bc_x.at(col);
            y[col] = bc_y.at(col);
        }
    }
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

int main(int argc, char **argv)
{
    DrawableTrimesh<> m(argv[1]);
    DrawableTrimesh<> m_ref = m;

    std::vector<uint>  boundary = m.get_ordered_boundary_vertices();
    std::vector<vec3d> circle   = n_sided_polygon(boundary.size(), CIRCLE);

    GLcanvas gui;
    gui.push(&m);
    gui.push(new SurfaceMeshControls<DrawableTrimesh<>>(&m,&gui));

    bool og_normals = true;
    int n_iters = 1;
    gui.callback_app_controls = [&]()
    {
        if(ImGui::SliderInt("Iters",&n_iters,1,100)){}

        if(ImGui::Checkbox("OG Normals",&og_normals))
        {
            if(og_normals)
            {
                for(uint pid=0; pid<m.num_polys(); ++pid)
                {
                    m.poly_data(pid).normal = m_ref.poly_data(pid).normal;
                }
                for(uint vid=0; vid<m.num_verts(); ++vid)
                {
                    m.vert_data(vid).normal = m_ref.vert_data(vid).normal;
                }
            }
            else
            {
                m.update_normals();
            }
            m.updateGL();
        }
    };

    gui.callback_key_pressed = [&](int key, int modifiers) -> bool
    {
        if(key==GLFW_KEY_I)
        {
            for(uint i=0; i<boundary.size(); ++i)
            {
                m.vert(boundary.at(i)) = circle.at(i);
            }
            m.updateGL();
            return true;
        }
        if(key==GLFW_KEY_SPACE)
        {
            laplacian(m,n_iters++);
            m.updateGL();
            gui.draw();
            return true;
        }
        if(key==GLFW_KEY_R)
        {
            m = m_ref;
            m.updateGL();
            return false;
        }
        if(key==GLFW_KEY_T)
        {
            m.copy_xyz_to_uvw(UVW_param);
            m.vector_verts() = m_ref.vector_verts();
            m.show_texture2D(TEXTURE_2D_ISOLINES,5.0);
            return true;
        }
        if(key==GLFW_KEY_U)
        {
            m.copy_uvw_to_xyz(UVW_param);
            m.updateGL();
            return true;
        }
        if(key==GLFW_KEY_L)
        {
            Eigen::SparseMatrix<double> L = laplacian_matrix3(m);
            std::map<uint,double> bc;
            uint nv = m.num_verts();
            for(uint i=0; i<boundary.size(); ++i)
            {
                uint vid = boundary.at(i);
                bc[     vid] = circle.at(i).x();
                bc[  nv+vid] = circle.at(i).y();
                bc[2*nv+vid] = circle.at(i).z();
            }

            Eigen::VectorXd xyz;
            Eigen::VectorXd rhs = Eigen::VectorXd::Zero(nv*3);
            p.push("solve3");
            solve_square_system_with_bc(L,rhs,xyz,bc);
            p.pop();

            for(uint vid=0; vid<m.num_verts(); ++vid)
            {
                m.vert(vid) = vec3d(xyz[     vid],
                                    xyz[  nv+vid],
                                    xyz[2*nv+vid]);
            }
            m.updateGL();
            return true;
        }
        if(key==GLFW_KEY_P)
        {
            Eigen::SparseMatrix<double> L = laplacian_matrix(m);

            std::map<uint,double> bc_x,bc_y;
            for(uint i=0; i<boundary.size(); ++i)
            {
                uint vid = boundary.at(i);
                bc_x[vid] = circle.at(i).x();
                bc_y[vid] = circle.at(i).y();
            }

            Eigen::VectorXd x,y;
            Eigen::VectorXd rhs_x = Eigen::VectorXd::Zero(m.num_verts());
            Eigen::VectorXd rhs_y = Eigen::VectorXd::Zero(m.num_verts());

            // p.push("solve");
            // solve_square_system_with_bc(L,rhs_x,x,bc_x);
            // solve_square_system_with_bc(L,rhs_y,y,bc_y);
            // p.pop();

            p.push("solve");
            solve(L,rhs_x,rhs_y,bc_x,bc_y,x,y);
            p.pop();

            for(uint vid=0; vid<m.num_verts(); ++vid)
            {
                m.vert(vid) = vec3d(x[vid],y[vid],0);
            }
            m.updateGL();
            return true;
        }
        return false;
    };

    gui.callback_mouse_left_click = [&](int modifiers) -> bool
    {
        if(modifiers & GLFW_MOD_SHIFT)
        {
            vec3d p;
            vec2d click = gui.cursor_pos();
            if(gui.unproject(click, p)) // transform click in a 3d point
            {
                uint vid = m.pick_vert(p);
                std::cout << "ID " << vid << std::endl;
                laplacian(m,vid);
                m.updateGL();
            }
        }
        return false;
    };

    return gui.launch();
}
