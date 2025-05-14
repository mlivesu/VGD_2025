#include <cinolib/gl/glcanvas.h>
#include <cinolib/gl/surface_mesh_controls.h>
#include <cinolib/meshes/meshes.h>
#include <cinolib/geometry/n_sided_poygon.h>
#include <cinolib/linear_solvers.h>

using namespace cinolib;

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

Eigen::SparseMatrix<double> laplacian_matrix(const DrawableTrimesh<> & m)
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
            Eigen::SparseMatrix<double> L = laplacian_matrix(m);
            Eigen::VectorXd xyz;
            Eigen::VectorXd rhs = Eigen::VectorXd::Zero(m.num_verts()*3);

            std::map<uint,double> bcs;
            uint nv = m.num_verts();
            for(uint i=0; i<boundary.size(); ++i)
            {
                uint vid = boundary.at(i);
                bcs[     vid] = circle.at(i).x();
                bcs[  nv+vid] = circle.at(i).y();
                bcs[2*nv+vid] = circle.at(i).z();
            }

            solve_square_system_with_bc(L,rhs,xyz,bcs);
            for(uint vid=0; vid<m.num_verts(); ++vid)
            {
                m.vert(vid) = vec3d(xyz[     vid],
                                    xyz[  nv+vid],
                                    xyz[2*nv+vid]);
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
