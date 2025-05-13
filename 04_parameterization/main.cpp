#include <cinolib/gl/glcanvas.h>
#include <cinolib/gl/surface_mesh_controls.h>
#include <cinolib/meshes/meshes.h>
#include <cinolib/geometry/n_sided_poygon.h>

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

int main(int argc, char **argv)
{
    DrawableTrimesh<> m(argv[1]);
    DrawableTrimesh<> m_ref = m;

    std::vector<uint>  boundary = m.get_ordered_boundary_vertices();
    std::vector<vec3d> circle   = n_sided_polygon(boundary.size(), CIRCLE);

    GLcanvas gui;
    gui.push(&m);
    gui.push(new SurfaceMeshControls<DrawableTrimesh<>>(&m,&gui));

    int n_iters = 1;
    gui.callback_app_controls = [&]()
    {
        if(ImGui::SliderInt("Iters",&n_iters,1,100)){}
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
