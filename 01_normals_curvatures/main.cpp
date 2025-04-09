#include <cinolib/gl/glcanvas.h>
#include <cinolib/gl/surface_mesh_controls.h>
#include <cinolib/drawable_segment_soup.h>

using namespace cinolib;

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

void compute_normals(const AbstractPolygonMesh<> & m,
                           DrawableSegmentSoup   & pn,
                           DrawableSegmentSoup   & vn)
{
    pn.clear();
    vn.clear();
    std::vector<vec3d> N(m.num_polys());
    double delta = m.bbox().diag()*0.05;

    for(uint pid=0; pid<m.num_polys(); ++pid)
    {
        vec3d v0  = m.poly_vert(pid,0);
        vec3d v1  = m.poly_vert(pid,1);
        vec3d v2  = m.poly_vert(pid,2);
        vec3d c   = (v0+v1+v2)/3.0;
        N.at(pid) = (v1-v0).cross(v2-v0);
        N.at(pid).normalize();
        pn.push_seg(c, c+N.at(pid)*delta);
    }

    for(uint vid=0; vid<m.num_verts(); ++vid)
    {
        vec3d n(0,0,0);
        for(uint pid : m.adj_v2p(vid)) n += N.at(pid);
        n.normalize();
        vn.push_seg(m.vert(vid), m.vert(vid)+n*delta);
    }
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

std::vector<double> mean_curvature(Trimesh<> & m)
{
    std::vector<double> H(m.num_verts(),0);
    for(uint vid=0; vid<m.num_verts(); ++vid)
    {
        vec3d Hn(0,0,0);
        for(uint nbr : m.adj_v2v(vid))
        {
            uint              eid = m.edge_id(nbr,vid);
            std::vector<uint> opp = m.verts_opposite_to(eid);
            assert(opp.size()==2);

            double w = 0.0;
            for(uint v_opp : opp)
            {
                vec3d  u     = m.vert(vid) - m.vert(v_opp);
                vec3d  v     = m.vert(nbr) - m.vert(v_opp);
                double alpha = acos(u.dot(v)/(u.norm()*v.norm()));
                w += cot(alpha);
            }
            Hn += w*(m.vert(vid) - m.vert(nbr));
        }
        double A = 0;
        for(uint pid : m.adj_v2p(vid)) A += m.poly_area(pid);
        Hn /= 2.0*A;
        H.at(vid) = Hn.norm();
    }
    return H;
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

std::vector<double> gaussian_curvature(Trimesh<> & m)
{
    std::vector<double> K(m.num_verts(),0);
    for(uint vid=0; vid<m.num_verts(); ++vid)
    {
        double A = 0;
        K.at(vid) = 2.0 * M_PI;
        for(uint pid : m.adj_v2p(vid))
        {
            uint   eid   = m.edge_opposite_to(pid,vid);
            uint   e0    = m.edge_vert_id(eid,0);
            uint   e1    = m.edge_vert_id(eid,1);
            vec3d  u     = m.vert(e0) - m.vert(vid);
            vec3d  v     = m.vert(e1) - m.vert(vid);
            double alpha = acos(u.dot(v)/(u.norm()*v.norm()));
            K.at(vid)   -= alpha;
            A += m.poly_area(pid)/3.0; // WARNING: approximated vertex area
        }
        K.at(vid) /= A;
    }
    return K;
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

int main(int argc, char **argv)
{
    DrawableTrimesh<> m(argv[1]);

    DrawableSegmentSoup vn,pn;
    vn.use_gl_lines = true;
    pn.use_gl_lines = true;
    vn.default_color = Color::RED();
    pn.default_color = Color::BLUE();
    compute_normals(m,pn,vn);

    std::vector<double> K = gaussian_curvature(m);
    std::vector<double> H = mean_curvature(m);
    double min_K = *std::min_element(K.begin(),K.end());
    double max_K = *std::max_element(K.begin(),K.end());
    double min_H = *std::min_element(H.begin(),H.end());
    double max_H = *std::max_element(H.begin(),H.end());

    GLcanvas gui;
    gui.push(&m);
    gui.push(new SurfaceMeshControls<DrawableTrimesh<>>(&m,&gui,"Mesh"));
    gui.show_side_bar = true;

    int vid = 0;
    bool show_v_normals = false;
    bool show_p_normals = false;

    gui.callback_app_controls = [&]()
    {
        if(ImGui::Checkbox("P Normals",&show_p_normals))
        {
            if(show_p_normals) gui.push(&pn);
            else               gui.pop(&pn);
        }
        if(ImGui::Checkbox("V Normals",&show_v_normals))
        {
            if(show_v_normals) gui.push(&vn);
            else               gui.pop(&vn);
        }
        if(ImGui::Button("Show K"))
        {
            // map K into [0,1], making sure that 0 maps to 0.5
            ScalarField s(K);
            double MIN = std::min(-fabs(min_K),-fabs(max_K));
            double MAX = std::max( fabs(min_K), fabs(max_K));
            for(uint vid=0; vid<m.num_verts(); ++vid) s[vid] = s[vid]-MIN/(MAX-MIN);
            s.copy_to_mesh(m);
            m.show_texture1D(TEXTURE_1D_HSV);
        }
        if(ImGui::Button("Show H"))
        {
            ScalarField s(H);
            s.normalize_in_01();
            s.copy_to_mesh(m);
            m.show_texture1D(TEXTURE_1D_HSV);
        }

        ImGui::Text("\n"
                    "H min = %f\n"
                    "H max = %f\n"
                    "\n"
                    "K min = %f\n"
                    "K max = %f\n"
                    "\n"
                    "Vert %d\n"
                    "K = %f\n"
                    "H = %f\n"
                    "\n\n",
                    min_H, max_H, min_K, max_K, vid, K.at(vid), H.at(vid));
    };

    gui.callback_mouse_left_click = [&](int modifiers) -> bool
    {
        if(modifiers & GLFW_MOD_SHIFT)
        {
            vec3d p;
            vec2d click = gui.cursor_pos();
            if(gui.unproject(click,p)) vid = m.pick_vert(p);
        }
        return false;
    };

    return gui.launch();
}
