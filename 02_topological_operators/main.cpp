#include <cinolib/gl/glcanvas.h>
#include <cinolib/gl/surface_mesh_controls.h>

using namespace cinolib;

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

void flood_chord(const DrawableQuadmesh<> & m,
                 const uint                 eid,
                       std::vector<uint>  & quads,
                       std::vector<uint>  & edges)
{
    quads.clear();
    edges.clear();

    std::vector<bool> visited(m.num_polys(),false);

    std::queue<uint> q;
    q.push(eid);
    edges.push_back(eid);

    while(!q.empty())
    {
        uint eid = q.front();
        q.pop();

        for(uint pid : m.adj_e2p(eid))
        {
            if(visited.at(pid)) continue;
            quads.push_back(pid);
            visited.at(pid) = true;
            uint next = m.edge_opposite_to(pid,eid);
            q.push(next);
            edges.push_back(next);
        }
    }
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

void collapse(DrawableQuadmesh<> & m, const std::vector<uint> & edges)
{

    std::vector<uint> v_map(m.num_verts());
    std::iota(v_map.begin(),v_map.end(),0); // identity map

    std::vector<uint> verts_to_process;
    for(uint eid : edges)
    {
        uint  e0  = m.edge_vert_id(eid,0);
        uint  e1  = m.edge_vert_id(eid,1);
        vec3d p   = (m.vert(e0) + m.vert(e1))*0.5;
        uint  vid = m.vert_add(p);
        v_map.at(e0) = vid;
        v_map.at(e1) = vid;
        verts_to_process.push_back(e0);
        verts_to_process.push_back(e1);
    }

    std::vector<uint> polys_to_remove;
    std::vector<uint> polys_visited(m.num_polys(),false);
    for(uint vid : verts_to_process)
    {
        for(uint pid : m.adj_v2p(vid))
        {
            if(polys_visited.at(pid)) continue;
            polys_visited.at(pid) = true;
            polys_to_remove.push_back(pid);

            if(m.poly_data(pid).color==Color::PASTEL_RED()) continue;

            std::vector<uint> V = m.poly_verts_id(pid);
            for(uint & v : V) v = v_map.at(v);
            m.poly_add(V);
        }
    }
    m.polys_remove(polys_to_remove);
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

int main(int argc, char **argv)
{
    DrawableQuadmesh<> m(argv[1]);

    GLcanvas gui;
    gui.push(&m);
    gui.push(new SurfaceMeshControls<DrawableQuadmesh<>>(&m,&gui));

    std::vector<uint> quads,edges;

    gui.callback_mouse_left_click = [&](int modifiers) -> bool
    {
        if(modifiers & GLFW_MOD_SHIFT)
        {
            vec3d p;
            vec2d click = gui.cursor_pos();
            if(gui.unproject(click, p))
            {
                uint eid = m.pick_edge(p);
                flood_chord(m,eid,quads,edges);
                m.poly_set_color(Color::WHITE());
                for(uint pid : quads)
                {
                    m.poly_data(pid).color = Color::PASTEL_RED();
                }
                m.updateGL();
            }
        }
        return false;
    };

    gui.callback_key_pressed = [&](int key, int modifier) -> bool
    {
        if(key==GLFW_KEY_C)
        {
            collapse(m,edges);
            m.updateGL();
            return true;
        }
        return false;
    };

    return gui.launch();
}
