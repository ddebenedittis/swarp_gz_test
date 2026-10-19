// Planar Coulomb floor friction with a uniform pressure distribution, applied as a wrench.
//
// DART (and bullet-featherstone) build friction from per-contact pyramids whose two axes
// are decoupled: a body sliding along one axis still has the full static friction of the
// other axis at every contact, which it uses to resist turning. Pushed off-centre with a
// flat face, the Push-T T then slides without turning, where real (isotropic) Coulomb
// friction turns it. This system replaces the T's floor friction (its floor contacts must be
// frictionless) with the exact limit surface of a uniform pressure over its footprint:
// every cell of a grid over the footprint gets -mu * p * v / max(|v|, v_eps).
//
//   <plugin filename="PlanarCoulombFriction" name="pusht_gz::PlanarCoulombFriction">
//     <mu>0.3</mu>               friction coefficient
//     <cell>0.02</cell>          grid spacing [m]
//     <v_eps>0.005</v_eps>       below this speed the friction is viscous (no chatter) [m/s]
//     <box>cx cy hx hy</box>     footprint boxes in the link frame, repeated, non-overlapping
//   </plugin>
//
// The link frame must sit on the floor (z = 0 at the footprint), which is where the
// friction acts; the normal load is the link's mass times |gravity|.

#include <cmath>
#include <sstream>
#include <memory>
#include <string>
#include <vector>

#include <gz/math/Vector3.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/World.hh>
#include <gz/sim/components/Inertial.hh>

namespace pusht_gz
{
class PlanarCoulombFriction : public gz::sim::System,
                              public gz::sim::ISystemConfigure,
                              public gz::sim::ISystemPreUpdate
{
public:
  void Configure(const gz::sim::Entity &_entity, const std::shared_ptr<const sdf::Element> &_sdf,
                 gz::sim::EntityComponentManager &_ecm, gz::sim::EventManager &) override
  {
    gz::sim::Model model(_entity);
    this->link = gz::sim::Link(model.CanonicalLink(_ecm));
    this->link.EnableVelocityChecks(_ecm, true);
    this->mu = _sdf->Get<double>("mu", 0.3).first;
    this->vEps = _sdf->Get<double>("v_eps", 0.005).first;
    const double cell = _sdf->Get<double>("cell", 0.02).first;
    for (auto e = _sdf->FindElement("box"); e; e = e->GetNextElement("box"))
    {
      std::istringstream in(e->Get<std::string>());
      double cx, cy, hx, hy;
      in >> cx >> cy >> hx >> hy;
      const int nx = std::max(1, static_cast<int>(std::round(2 * hx / cell)));
      const int ny = std::max(1, static_cast<int>(std::round(2 * hy / cell)));
      for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
        {
          // Weight each cell by its area, so boxes of different spacing stay uniform.
          this->px.push_back(cx - hx + (i + 0.5) * 2 * hx / nx);
          this->py.push_back(cy - hy + (j + 0.5) * 2 * hy / ny);
          this->area.push_back(4 * hx * hy / (nx * ny));
        }
    }
    double total = 0;
    for (double a : this->area)
      total += a;
    for (double &a : this->area)
      a /= total;
    const auto inertial = _ecm.Component<gz::sim::components::Inertial>(this->link.Entity());
    const double mass = inertial ? inertial->Data().MassMatrix().Mass() : 1.0;
    const auto gravity = gz::sim::World(gz::sim::worldEntity(_ecm)).Gravity(_ecm);
    this->load = mass * (gravity ? gravity->Length() : 9.81);
    gzmsg << "PlanarCoulombFriction: " << this->px.size() << " cells, mu " << this->mu
          << ", normal load " << this->load << " N\n";
  }

  void PreUpdate(const gz::sim::UpdateInfo &_info, gz::sim::EntityComponentManager &_ecm) override
  {
    if (_info.paused || this->px.empty())
      return;
    const auto pose = this->link.WorldPose(_ecm);
    const auto v = this->link.WorldLinearVelocity(_ecm);
    const auto w = this->link.WorldAngularVelocity(_ecm);
    if (!pose || !v || !w)
      return;
    const double yaw = pose->Rot().Yaw();
    const double c = std::cos(yaw), s = std::sin(yaw);
    double fx = 0, fy = 0, tz = 0;
    for (size_t k = 0; k < this->px.size(); ++k)
    {
      const double rx = c * this->px[k] - s * this->py[k];
      const double ry = s * this->px[k] + c * this->py[k];
      const double vx = v->X() - w->Z() * ry;
      const double vy = v->Y() + w->Z() * rx;
      const double scale = this->mu * this->load * this->area[k] / std::max(std::hypot(vx, vy), this->vEps);
      const double gx = -scale * vx, gy = -scale * vy;
      fx += gx;
      fy += gy;
      tz += rx * gy - ry * gx;
    }
    // Force at the link origin (on the floor), torque about the vertical through it.
    this->link.AddWorldWrench(_ecm, gz::math::Vector3d(fx, fy, 0), gz::math::Vector3d(0, 0, tz),
                              gz::math::Vector3d::Zero);
  }

private:
  gz::sim::Link link;
  double mu{0.3}, vEps{0.005}, load{0};
  std::vector<double> px, py, area;
};
}  // namespace pusht_gz

GZ_ADD_PLUGIN(pusht_gz::PlanarCoulombFriction, gz::sim::System,
              pusht_gz::PlanarCoulombFriction::ISystemConfigure,
              pusht_gz::PlanarCoulombFriction::ISystemPreUpdate)
