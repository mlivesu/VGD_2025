#include <cinolib/meshes/drawable_trimesh.h>
#include <cinolib/gl/glcanvas.h>
#include <cinolib/gl/surface_mesh_controls.h>
#include <cinolib/geometry/n_sided_poygon.h>
#include <cinolib/linear_solvers.h>
#include <cinolib/profiler.h>
#include <cinolib/lscm.h>

using namespace cinolib;

Profiler prof;

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

void distortion(const DrawableTrimesh<>   & m_xyz,
                const DrawableTrimesh<>   & m_uv,
                std::vector<double> & conf_dist,
                std::vector<double> & ARAP_dist)
{
    conf_dist.resize(m_xyz.num_polys());
    ARAP_dist.resize(m_xyz.num_polys());

    for(uint pid=0; pid<m_xyz.num_polys(); ++pid)
    {
        vec3d A0 = m_xyz.poly_vert(pid,0);
        vec3d A1 = m_xyz.poly_vert(pid,1);
        vec3d A2 = m_xyz.poly_vert(pid,2);

        vec3d Tu =  A1-A0;
        vec3d N  = (A1-A0).cross(A2-A0);
        vec3d Tv = Tu.cross(N);
        Tu.normalize();
        Tv.normalize();

        vec2d a0(0,0);
        vec2d a1(A1.dist(A0),0);
        vec2d a2((A2-A0).dot(Tu),(A2-A0).dot(Tv));

        vec2d b0 = m_uv.poly_vert(pid,0).rem_coord();
        vec2d b1 = m_uv.poly_vert(pid,1).rem_coord();
        vec2d b2 = m_uv.poly_vert(pid,2).rem_coord();

        mat2d uv0({a1-a0, a2-a0});
        mat2d uv1({b1-b0, b2-b0});
        mat2d J = uv1 * uv0.inverse();

        vec2d S;
        mat2d U,V;
        J.SVD(U,S,V);

        conf_dist.at(pid) = (S[0]-S[1])*(S[0]-S[1]);
        ARAP_dist.at(pid) = (S[0]-1)*(S[0]-1) + (S[1]-1)*(S[1]-1);
    }
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

void laplacian(DrawableTrimesh<> & m, const uint vid)
{
    vec3d p(0,0,0);
    for(uint nbr : m.adj_v2v(vid))
    {
        p += m.vert(nbr);
    }
    m.vert(vid) = p / m.vert_valence(vid);
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

void laplacian_all(DrawableTrimesh<> & m, const uint n_iter)
{
    for(uint i=0; i<n_iter; ++i)
    {
        for(uint vid=0; vid<m.num_verts(); ++vid)
        {
            if(m.vert_is_boundary(vid)) continue;
            laplacian(m,vid);
        }
    }
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

Eigen::SparseMatrix<double> laplacian3(const DrawableTrimesh<> & m)
{
    std::vector<Eigen::Triplet<double>> entries;
    uint nv = m.num_verts();
    for(uint vid=0; vid<m.num_verts(); ++vid)
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

    Eigen::SparseMatrix<double> L(3*nv,3*nv);
    L.setFromTriplets(entries.begin(),entries.end());
    return L;
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

Eigen::SparseMatrix<double> laplacian(const DrawableTrimesh<> & m)
{
    std::vector<Eigen::Triplet<double>> entries;
    uint nv = m.num_verts();
    for(uint vid=0; vid<m.num_verts(); ++vid)
    {
        entries.emplace_back(vid,vid,double(m.vert_valence(vid)));

        for(uint nbr : m.adj_v2v(vid))
        {
            entries.emplace_back(vid,nbr,-1);
        }
    }

    Eigen::SparseMatrix<double> L(nv,nv);
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
    std::vector<vec3d> circle   = n_sided_polygon(boundary.size(),CIRCLE);

    GLcanvas gui;
    gui.push(&m);
    gui.push(new SurfaceMeshControls<DrawableTrimesh<>>(&m,&gui));

    std::vector<double> conf_dist;
    std::vector<double> ARAP_dist;

    int  tot_iters = 0;
    int  iters = 1;
    gui.callback_key_pressed = [&](int key, int modifiers) -> bool
    {
        if(key==GLFW_KEY_SPACE)
        {
            prof.push("Iterative");
            laplacian_all(m,iters++);
            prof.pop();
            m.updateGL();
            tot_iters += iters;
            std::cout << "#iters " << tot_iters << std::endl;
            return true;
        }
        if(key==GLFW_KEY_I)
        {
            for(uint i=0; i<boundary.size(); ++i)
            {
                m.vert(boundary.at(i)) = circle.at(i);
            }
            m.updateGL();
        }
        if(key==GLFW_KEY_T)
        {
            m.copy_xyz_to_uvw(UVW_param);
            m.vector_verts() = m_ref.vector_verts();
            m.update_normals();
            m.show_texture2D(TEXTURE_2D_ISOLINES, 5.0);
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
            Eigen::SparseMatrix<double> L = laplacian3(m);

            uint nv = m.num_verts();
            std::map<uint,double> bc;
            for(uint i=0; i<boundary.size(); ++i)
            {
                uint vid = boundary.at(i);
                bc[     vid] = circle.at(i).x();
                bc[  nv+vid] = circle.at(i).y();
                bc[2*nv+vid] = circle.at(i).z();
            }

            Eigen::VectorXd xyz;
            Eigen::VectorXd rhs = Eigen::VectorXd::Zero(3*nv);

            prof.push("solve_square_system_with_bc");
            solve_square_system_with_bc(L, rhs, xyz, bc);
            prof.pop();

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
            Eigen::SparseMatrix<double> L = laplacian(m);

            uint nv = m.num_verts();
            std::map<uint,double> bc_x, bc_y;
            for(uint i=0; i<boundary.size(); ++i)
            {
                uint vid = boundary.at(i);
                bc_x[vid] = circle.at(i).x();
                bc_y[vid] = circle.at(i).y();
            }

            Eigen::VectorXd x,y;
            Eigen::VectorXd rhs_x = Eigen::VectorXd::Zero(nv);
            Eigen::VectorXd rhs_y = Eigen::VectorXd::Zero(nv);

            prof.push("solve_square_system_with_bc (only X and Y");
            solve_square_system_with_bc(L, rhs_x, x, bc_x);
            solve_square_system_with_bc(L, rhs_y, y, bc_y);
            prof.pop();

            prof.push("solve_square_system_with_bc (only X and Y - factorized once");
            solve(L,rhs_x,rhs_y,bc_x,bc_y,x,y);
            prof.pop();

            for(uint vid=0; vid<m.num_verts(); ++vid)
            {
                m.vert(vid) = vec3d(x[vid],
                                    y[vid],
                                    0);
            }
            m.updateGL();
            return true;
        }
        if(key==GLFW_KEY_B)
        {
            std::map<uint,vec2d> bc;
            bc[boundary.front()] = vec2d(-1,-1);
            bc[boundary.at(boundary.size()/2)] = vec2d(1,1);
            ScalarField uv = LSCM(m,bc);
            uv.copy_to_mesh(m,UV_param);
            m.copy_uvw_to_xyz(UV_param);
            for(uint vid=0; vid<m.num_verts(); ++vid) m.vert(vid).z() = 0.0;
            //m.normalize_bbox();
            m.update_normals();
            m.updateGL();
        }
        return false;
    };

    gui.callback_mouse_left_click = [&](int modifiers) -> bool
    {
        if(modifiers & GLFW_MOD_SHIFT)
        {
            vec3d p;
            vec2d click = gui.cursor_pos();
            if(gui.unproject(click,p))
            {
                uint vid = m.pick_vert(p);
                laplacian(m,vid);
                m.updateGL();
            }
        }
        return false;
    };

    gui.callback_app_controls = [&]()
    {
        if(ImGui::Button("ARAP dist"))
        {
            if(ARAP_dist.empty()) distortion(m_ref,m,conf_dist,ARAP_dist);

            double E_ARAP = 0;
            for(uint pid=0; pid<m.num_polys(); ++pid)
            {
                E_ARAP += ARAP_dist.at(pid) * m.poly_area(pid);
                m.poly_data(pid).color = Color::red_ramp_01(std::min(ARAP_dist.at(pid),50.0));
                m_ref.poly_data(pid).color = m.poly_data(pid).color;
            }
            m_ref.show_poly_color();
            m.show_poly_color();
            std::cout << "E_ARAP : " << E_ARAP << std::endl;
        }
        if(ImGui::Button("Conformal dist"))
        {
            if(conf_dist.empty()) distortion(m_ref,m,conf_dist,ARAP_dist);

            double E_conf = 0;
            for(uint pid=0; pid<m.num_polys(); ++pid)
            {
                E_conf += conf_dist.at(pid) * m.poly_area(pid);;
                m.poly_data(pid).color = Color::red_ramp_01(std::min(conf_dist.at(pid),50.0));
                m_ref.poly_data(pid).color = m.poly_data(pid).color;
            }
            m_ref.show_poly_color();
            m.show_poly_color();
            std::cout << "E_conf : " << E_conf << std::endl;
        }
    };

    return gui.launch();
}
