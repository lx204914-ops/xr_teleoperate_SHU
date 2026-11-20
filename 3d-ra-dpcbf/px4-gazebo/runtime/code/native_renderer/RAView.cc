#include <gz/gui/Plugin.hh>
#include <gz/gui/Application.hh>
#include <gz/gui/MainWindow.hh>
#include <gz/gui/GuiEvents.hh>
#include <gz/rendering/RenderingIface.hh>
#include <gz/rendering/RenderEngine.hh>
#include <gz/rendering/Scene.hh>
#include <gz/rendering/Visual.hh>
#include <gz/rendering/Marker.hh>
#include <gz/rendering/Material.hh>
#include <gz/rendering/MeshDescriptor.hh>
#include <gz/rendering/Sensor.hh>
#include <gz/common/MeshManager.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/marker_v.pb.h>
#include <gz/msgs/double_v.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <gz/plugin/Register.hh>
#include <tinyxml2.h>
#include <QEvent>
#include <QString>
#include <map>
#include <mutex>
#include <atomic>
#include <sstream>
#include <cmath>
#include <chrono>
#include <thread>
#include <algorithm>
#include <vector>
#include <fstream>
#include <cstdint>

using V=gz::math::Vector3d;
using C=gz::math::Color;
namespace R=gz::rendering;

class RAView final:public gz::gui::Plugin {
 Q_OBJECT
 Q_PROPERTY(QString status READ Status NOTIFY StatusChanged)
 struct Entry {R::VisualPtr visual;R::MarkerPtr marker;R::MaterialPtr material;};

 struct Param {int mode=0;V w;V e;double vertex=0,lambda=0,h=0,R=0,Vr=0,ref=0,radius=0,omega=0,stamp=0,base=0,span=1,scale=.6,cap=1.5,Tr=.8,beta=0,clearance=0,relativeTime=0;C color;};
 struct Motion {V position,velocity;double radius=0;};
 std::map<std::string,Entry> entries;std::map<int,Param> parameters;std::map<int,Motion> motions;
 double worldTime=0,maximumFormulaError=0,maximumAttachmentError=0,maximumArrowAnchorError=0,minimumArrowAlignment=1;
 double minimumRadius=0,maximumRadius=0;unsigned int attachedBalls=0;

 gz::transport::Node node;gz::transport::Node::Publisher statusPub,frameAuditPub;
 std::map<int,std::vector<V>> meshProbes;std::map<int,V> riskEdgeProbes,ballCenterProbes;unsigned long renderFrames=0;
 std::mutex mutex;gz::msgs::Double_V lastStatus;std::ofstream auditFile;std::string auditPath;gz::msgs::Marker_V latest;bool dirty=false,haveTruth=false,replicaBuilt=false,completionPending=false,terminalFrameAudited=false;
 gz::math::Pose3d dronePose;std::map<int,V> actualObstacles;
 std::map<double,gz::math::Pose3d> truthFrames;std::map<double,std::map<int,Motion>> motionFrames;
 R::ScenePtr scene;R::VisualPtr drone;std::string basePath;std::map<int,R::VisualPtr> ballCache;
 bool geometryCached=false; double lastGeometryTime=0; V lastGeometryCamera;
 std::chrono::steady_clock::time_point nextDisplayFrame{};
 std::atomic<double> time{-1};std::atomic<int> phase{0};int completionOutcome=1;unsigned long frames=0;
 public:
 QString Status()const {if(time<0)return "RA-DPCBF  |  waiting for controller";return QString("%3  |  %1  |  t = %2 s  |  relative-velocity schematic (zero origin)").arg(phase==1?"ARRIVED":(phase==2?"PREPARING":(phase==3?"STOPPED":"LIVE"))).arg(time.load(),0,'f',1).arg(qEnvironmentVariable("CBF_METHOD_LABEL","RA-DPCBF"));}
 signals:void StatusChanged();
 protected:
 void LoadConfig(const tinyxml2::XMLElement* xml)override{
   this->title="RAView";
   if(xml&&xml->FirstChildElement("model_path"))basePath=xml->FirstChildElement("model_path")->GetText();
   gz::gui::App()->findChild<gz::gui::MainWindow*>()->installEventFilter(this);
   node.Subscribe("/ra/native_geometry",&RAView::Geometry,this);
   node.Advertise("/ra/native_geometry_rpc",&RAView::GeometryService,this);
   node.Subscribe("/ra/vehicle_truth",&RAView::Truth,this);
   node.Subscribe("/ra/obstacle_actual",&RAView::Obstacles,this);
   node.Subscribe("/ra/obstacle_kinematics",&RAView::Kinematics,this);
   statusPub=node.Advertise<gz::msgs::Double_V>("/ra/native_geometry_status");
   frameAuditPub=node.Advertise<gz::msgs::Double_V>("/ra/native_frame_audit");
 }
 void Geometry(const gz::msgs::Marker_V&msg){std::lock_guard<std::mutex> lock(mutex);if(msg.marker_size()==0){for(const auto&d:msg.header().data())if(d.key()=="completed"){completionPending=true;completionOutcome=1;}else if(d.key()=="failed"){completionPending=true;completionOutcome=3;}return;}latest=msg;dirty=true;}
 bool GeometryService(const gz::msgs::Marker_V&request,gz::msgs::Double_V&response){
   Geometry(request);std::lock_guard<std::mutex>lock(mutex);
   for(const auto&d:request.header().data())if(d.key()=="audit_path"&&d.value_size()&&auditPath!=d.value(0)){
     auditPath=d.value(0);auditFile.close();auditFile.open(auditPath,std::ios::binary|std::ios::trunc);
   }
   response=lastStatus;return true;
 }
 void Truth(const gz::msgs::Double_V&msg){if(msg.data_size()!=14)return;std::lock_guard<std::mutex>lock(mutex);truthFrames[msg.data(0)]=gz::math::Pose3d(V(msg.data(1),msg.data(2),msg.data(3)),gz::math::Quaterniond(msg.data(13),msg.data(10),msg.data(11),msg.data(12)));while(truthFrames.size()>128)truthFrames.erase(truthFrames.begin());}
 void Obstacles(const gz::msgs::Pose_V&msg){std::lock_guard<std::mutex>lock(mutex);for(const auto&p:msg.pose()){auto at=p.name().rfind('_');if(at!=std::string::npos){try{actualObstacles[std::stoi(p.name().substr(at+1))]=V(p.position().x(),p.position().y(),p.position().z());}catch(...){}}}}

 void Kinematics(const gz::msgs::Double_V&msg){
   if(msg.data_size()<2)return;int n=int(msg.data(1));if(msg.data_size()!=2+10*n)return;
   std::lock_guard<std::mutex>lock(mutex);
   std::map<int,Motion> sample;for(int j=0;j<n;j++){int k=2+10*j;sample[j]={V(msg.data(k),msg.data(k+1),msg.data(k+2)),V(msg.data(k+3),msg.data(k+4),msg.data(k+5)),msg.data(k+9)};}motionFrames[msg.data(0)]=std::move(sample);while(motionFrames.size()>128)motionFrames.erase(motionFrames.begin());
 }
 void Init(){auto names=R::loadedEngines();if(names.empty())return;auto engine=R::engine(names[0]);if(!engine||!engine->SceneCount())return;scene=engine->SceneByIndex(0);if(!scene||!scene->IsInitialized()){scene.reset();return;}}
 Entry& Make(const std::string&key,R::MarkerType type,C color,double alpha,bool lighting=false){
   auto it=entries.find(key);if(it!=entries.end())return it->second;
   Entry en;en.visual=scene->CreateVisual("ra_view_"+key);en.marker=scene->CreateMarker();en.marker->SetType(type);en.marker->SetSize(2.0);
   en.material=scene->CreateMaterial("ra_mat_"+key);en.material->SetAmbient(color);en.material->SetDiffuse(color);en.material->SetEmissive(C(color.R()*.18,color.G()*.18,color.B()*.18,1));
   en.material->SetLightingEnabled(lighting);en.material->SetTransparency(1-alpha);en.material->SetDepthWriteEnabled(alpha>.98);en.material->SetCastShadows(false);en.material->SetReceiveShadows(false);
   en.marker->SetMaterial(en.material,false);en.visual->AddGeometry(en.marker);scene->RootVisual()->AddChild(en.visual);
   return entries.emplace(key,std::move(en)).first->second;
 }
 void Apply(const gz::msgs::Marker&m){

   if(m.ns()=="ra_parameters"){
     if(m.point_size()!=9)return;auto&a=m.point(0);auto&b=m.point(1);auto&r=m.point(2);auto&t=m.point(3);auto&base=m.point(4);auto&pos=m.point(5);auto&vel=m.point(6);auto&shift=m.point(7);auto&c=m.material().diffuse();
     Param p;p.mode=int(m.scale().z());p.w=V(m.point(8).x(),m.point(8).y(),m.point(8).z());p.e=V(a.x(),a.y(),a.z()).Normalized();p.Tr=shift.x();p.beta=shift.y();p.relativeTime=shift.z();p.vertex=b.x();p.lambda=b.y();p.h=b.z();p.R=r.x();p.Vr=r.y();p.ref=r.z();p.radius=t.x();p.omega=t.y();p.stamp=t.z();p.base=base.x();p.span=base.y();p.color=C(c.r(),c.g(),c.b(),1);p.scale=m.scale().x();p.cap=m.scale().y();p.clearance=base.z();parameters[int(m.id())]=p;
     if(!motions.count(int(m.id())))motions[int(m.id())]={V(pos.x(),pos.y(),pos.z()),V(vel.x(),vel.y(),vel.z()),p.radius};return;
   }

   R::MarkerType type=R::MarkerType::MT_TRIANGLE_LIST;
   auto&c=m.material().diffuse();C color(c.r(),c.g(),c.b(),1);auto&en=Make(m.ns()+"_"+std::to_string(m.id()),type,color,c.a());
   en.material->SetDiffuse(color);en.material->SetAmbient(color);en.material->SetTransparency(1-c.a());en.material->SetDepthWriteEnabled(c.a()>.98);
   en.marker->ClearPoints();
   if(m.type()==gz::msgs::Marker::TRIANGLE_LIST){
     for(const auto&p:m.point())en.marker->AddPoint(V(p.x(),p.y(),p.z()),color);
   }else{
     V camera(9,-14,12);if(scene->SensorCount())camera=scene->SensorByIndex(0)->WorldPose().Pos();
     const bool strip=m.type()==gz::msgs::Marker::LINE_STRIP;
     const double width=m.ns()=="profile"?.032:m.ns()=="rim"?.022:.025;
     for(int k=0;k+1<m.point_size();k+=strip?1:2){
       const auto&a=m.point(k);const auto&b=m.point(k+1);V p(a.x(),a.y(),a.z()),q(b.x(),b.y(),b.z());
       V side=(q-p).Cross(camera-(p+q)*.5);if(side.Length()<1e-10)continue;side.Normalize();side*=width*.5;
       C pc=color;if(m.materials_size()==m.point_size()){auto&cc=m.materials(k).diffuse();pc=C(cc.r(),cc.g(),cc.b(),1);}
       for(auto&point:std::vector<V>{p-side,q-side,q+side,p-side,q+side,p+side,q+side,q-side,p-side,p+side,q+side,p-side})en.marker->AddPoint(point,pc);
     }
   }
   en.visual->SetWorldPose(gz::math::Pose3d(V(m.pose().position().x(),m.pose().position().y(),m.pose().position().z()),gz::math::Quaterniond::Identity));
 }
 static gz::math::Pose3d Pose(const tinyxml2::XMLElement*e){double x=0,y=0,z=0,r=0,p=0,a=0;if(e&&e->GetText()){std::istringstream s(e->GetText());s>>x>>y>>z>>r>>p>>a;}return gz::math::Pose3d(x,y,z,r,p,a);}
 void Drone(){
   if(!replicaBuilt&&!basePath.empty()){
     tinyxml2::XMLDocument doc;if(doc.LoadFile((basePath+"/model.sdf").c_str())==tinyxml2::XML_SUCCESS){
       auto*model=doc.FirstChildElement("sdf")->FirstChildElement("model");drone=scene->CreateVisual("ra_display_x500");scene->RootVisual()->AddChild(drone);
       auto mat=scene->CreateMaterial("ra_x500_grey");mat->SetAmbient(.35,.38,.4,1);mat->SetDiffuse(.35,.38,.4,1);mat->SetSpecular(.2,.2,.2,1);mat->SetCastShadows(true);
       int count=0;for(auto*link=model->FirstChildElement("link");link;link=link->NextSiblingElement("link")){
         auto lp=Pose(link->FirstChildElement("pose"));for(auto*vis=link->FirstChildElement("visual");vis;vis=vis->NextSiblingElement("visual")){
           auto*g=vis->FirstChildElement("geometry");auto*mesh=g?g->FirstChildElement("mesh"):nullptr;if(!mesh||!mesh->FirstChildElement("uri"))continue;
           std::string uri=mesh->FirstChildElement("uri")->GetText();const std::string prefix="model://x500_base";if(uri.rfind(prefix,0)!=0)continue;uri=basePath+uri.substr(prefix.size());
           auto loaded=gz::common::MeshManager::Instance()->Load(uri);if(!loaded)continue;
           R::MeshDescriptor descriptor;descriptor.mesh=loaded;auto geometry=scene->CreateMesh(descriptor);if(!geometry)continue;auto v=scene->CreateVisual("ra_x500_part_"+std::to_string(count++));v->SetLocalPose(lp*Pose(vis->FirstChildElement("pose")));
           double sx=1,sy=1,sz=1;auto*scale=mesh->FirstChildElement("scale");if(scale&&scale->GetText()){std::istringstream s(scale->GetText());s>>sx>>sy>>sz;}v->SetLocalScale(sx,sy,sz);geometry->SetMaterial(mat,false);v->AddGeometry(geometry);drone->AddChild(v);
         }
       }
       drone->SetLocalScale(1.7,1.7,1.7);replicaBuilt=count>0;
       auto original=scene->VisualByName("x500_0");if(original&&replicaBuilt)original->SetVisible(false);
     }
   }
   if(drone&&haveTruth)drone->SetWorldPose(dronePose);
 }

 V Camera()const {return scene->SensorCount()?scene->SensorByIndex(0)->WorldPose().Pos():V(9,-14,12);}
 void Ribbon(const std::string&key,const std::vector<V>&points,const V&origin,C color,double alpha,double width,bool strip=true){
   auto&en=Make(key,R::MarkerType::MT_TRIANGLE_LIST,color,alpha);en.material->SetTransparency(1-alpha);en.marker->ClearPoints();V camera=Camera();
   for(std::size_t k=0;k+1<points.size();k+=strip?1:2){V p=points[k],q=points[k+1];V side=(q-p).Cross(camera-origin-(p+q)*.5);if(side.Length()<1e-12)continue;side.Normalize();side*=width*.5;
     for(auto&x:std::vector<V>{p-side,q-side,q+side,p-side,q+side,p+side,q+side,q-side,p-side,p+side,q+side,p-side})en.marker->AddPoint(x,color);
   }
   en.visual->SetWorldPose(gz::math::Pose3d(origin,gz::math::Quaterniond::Identity));
 }
 void Boundaries(bool rebuild){
   if(!haveTruth)return;V origin=dronePose.Pos();
   for(auto&[id,p]:parameters){
     if(p.lambda<0||p.scale<=0)continue;const std::string key=std::to_string(id);
     if(p.mode==1||p.mode==2){
       std::vector<V>velocity;if(p.w.Length()>1e-9){V d=p.w.Normalized();V end=.9*d;V side=d.Cross(Camera()-origin);if(side.Length()<1e-8)side=d.Cross(V::UnitZ);side.Normalize();velocity={V::Zero,end,end,end-.14*d+.06*side,end,end-.14*d-.06*side};}
       Ribbon("relative_velocity_"+key,velocity,origin,p.color,.95,.025,false);
       if(p.mode==2){
         // Fixed display size: direction schematic, not the exact C3BF safe set.
         V n=p.e.Cross(V::UnitZ);if(n.Length()<1e-8)n=p.e.Cross(V::UnitY);n.Normalize();V b=p.e.Cross(n);V center=-.9*p.e;double radius=.3;
         std::vector<V>rim,spokes;for(int j=0;j<=48;j++){double a=2*M_PI*j/48;rim.push_back(center+radius*(std::cos(a)*n+std::sin(a)*b));}
         for(int j=0;j<6;j++){double a=2*M_PI*j/6;spokes.push_back(V::Zero);spokes.push_back(center+radius*(std::cos(a)*n+std::sin(a)*b));}
         Ribbon("cone_rim_"+key,rim,origin,p.color,.24,.012);Ribbon("cone_side_"+key,spokes,origin,p.color,.20,.012,false);
       }
       meshProbes[id]={V::Zero,V::Zero,V::Zero};continue;
     }
     if(p.mode==3){
       V tangent=p.e.Cross(Camera()-origin);if(tangent.Length()<1e-8)tangent=p.e.Cross(V::UnitZ);if(tangent.Length()<1e-8)tangent=p.e.Cross(V::UnitY);tangent.Normalize();
       std::vector<V>curve;for(int j=0;j<=100;j++){double z=p.cap*(2.*j/100-1);V point=p.scale*(-(p.vertex+p.lambda*z*z)*p.e+z*tangent);curve.push_back(point);}
       Ribbon("park_profile_"+key,curve,origin,p.color,.85,.035);meshProbes[id]={curve.front(),curve[50],curve.back()};
       for(V point:meshProbes[id]){V w=point/p.scale;double q=w.Dot(p.e);maximumFormulaError=std::max(maximumFormulaError,std::abs(q+p.vertex+p.lambda*(w-q*p.e).SquaredLength()));}
       continue;
     }
     double cap=p.cap; // one fixed transverse velocity radius for every obstacle/frame
     V n=p.e.Cross(V::UnitZ);if(n.Length()<1e-8)n=p.e.Cross(V::UnitY);n.Normalize();V n2=p.e.Cross(n);
     double focus=std::exp(-std::pow(std::max(p.clearance,0.)/5.,2)); double alpha=.003+.045*focus;
     auto&surface=Make("surface_"+key,R::MarkerType::MT_TRIANGLE_LIST,p.color,alpha);surface.material->SetTransparency(1-alpha);
     auto point=[&](double r,double t){return p.scale*(-(p.vertex+p.lambda*r*r)*p.e+r*std::cos(t)*n+r*std::sin(t)*n2);};
     if(rebuild){
       surface.marker->ClearPoints();meshProbes[id].clear();
       for(int j=0;j<24;j++)for(int k=0;k<8;k++){
         double t0=2*M_PI*j/24,t1=2*M_PI*(j+1)/24,r0=cap*k/8,r1=cap*(k+1)/8;
         V a=point(r0,t0),b=point(r1,t0),c=point(r1,t1),d=point(r0,t1);
         for(auto&x:std::vector<V>{a,b,c,a,c,d,c,b,a,d,c,a})surface.marker->AddPoint(x,p.color);
         if(j==0&&k==0)meshProbes[id].push_back(a);if(j==0&&k==7)meshProbes[id].push_back(b);if(j==6&&k==3)meshProbes[id].push_back(c);
         V w=c/p.scale;double axial=w.Dot(p.e);double residual=axial+p.vertex+p.lambda*(w-p.e*axial).SquaredLength();maximumFormulaError=std::max(maximumFormulaError,std::abs(residual));
       }
     }
     surface.visual->SetWorldPose(gz::math::Pose3d(origin,gz::math::Quaterniond::Identity));
     maximumAttachmentError=std::max(maximumAttachmentError,(surface.visual->WorldPose().Pos()-origin).Length());
     std::vector<V>rim;for(int j=0;j<=80;j++)rim.push_back(point(cap,2*M_PI*j/80));Ribbon("rim_"+key,rim,origin,p.color,.025+.30*focus,.016);
     V tangent=p.e.Cross(Camera()-origin);if(tangent.Length()<1e-8)tangent=n;else tangent.Normalize();
     std::vector<V>profile;for(int j=0;j<=100;j++){double r=cap*(2.*j/100-1);profile.push_back(p.scale*(-(p.vertex+p.lambda*r*r)*p.e+r*tangent));}
     Ribbon("profile_"+key,profile,origin,p.color,.08+.90*focus,.04);
   }
 }

 R::VisualPtr Ball(int id){
   auto cached=ballCache.find(id);if(cached!=ballCache.end())return cached->second; std::string name="ra_obstacle_"+std::to_string(id);auto ball=scene->VisualByName(name);if(ball){ballCache[id]=ball;return ball;}
   for(unsigned int i=0;i<scene->VisualCount();i++){auto v=scene->VisualByIndex(i);auto n=v->Name();if(n.size()>name.size()+2&&n.substr(n.size()-name.size()-2)=="::"+name){ballCache[id]=v;return v;}}
   return nullptr;
 }
 void AttachBall(R::VisualPtr visual,int id){
   auto ball=Ball(id);if(!ball)return;
   if(visual->Parent()!=ball){auto parent=visual->Parent();if(parent)parent->RemoveChild(visual);ball->AddChild(visual);}
   visual->SetLocalPose(gz::math::Pose3d(V::Zero,gz::math::Quaterniond::Identity));
 }
 void HalosAndVelocity(){
   minimumRadius=1e9;maximumRadius=0;attachedBalls=0;
   for(auto&[id,p]:parameters){
     auto it=motions.find(id);if(it==motions.end())continue;auto motion=it->second;V center=motion.position;
     auto actual=actualObstacles.find(id);if(actual!=actualObstacles.end())center=actual->second;auto ball=Ball(id);if(ball){center=ball->WorldPose().Pos();attachedBalls++;}
     double outer=p.R; // same actual control snapshot as the surface, no cross-stamp extrapolation
     minimumRadius=std::min(minimumRadius,outer);maximumRadius=std::max(maximumRadius,outer);
     V normal=(Camera()-center).Normalized(),u=normal.Cross(V::UnitZ);if(u.Length()<1e-8)u=normal.Cross(V::UnitY);u.Normalize();V v=normal.Cross(u);
     double risk=std::clamp((outer-p.base)/p.span,0.,1.);
     if(p.mode==0){
     for(int band=0;band<6;band++){
       double f0=band/6.,f1=(band+1)/6.,mid=(f0+f1)*.5;
       double alpha=(.025+.075*risk)*(1-mid)*(1-mid);
       auto&en=Make("halo_"+std::to_string(id)+"_"+std::to_string(band),R::MarkerType::MT_TRIANGLE_LIST,C(.58,.43,.78,1),alpha);en.material->SetTransparency(1-alpha);en.marker->ClearPoints();
       double r0=p.radius+(outer-p.radius)*f0,r1=p.radius+(outer-p.radius)*f1;
       for(int j=0;j<48;j++){
         double a=2*M_PI*j/48,c=2*M_PI*(j+1)/48;V d0=u*std::cos(a)+v*std::sin(a),d1=u*std::cos(c)+v*std::sin(c);
         V a0=r0*d0,b0=r1*d0,c0=r1*d1,d0p=r0*d1;
         for(auto&x:std::vector<V>{a0,b0,c0,a0,c0,d0p,c0,b0,a0,d0p,c0,a0})en.marker->AddPoint(x,C(.58,.43,.78,1));
       }
       en.visual->SetWorldPosition(center);AttachBall(en.visual,id);
     }
     std::vector<V>edge;for(int j=0;j<=96;j++){double t=2*M_PI*j/96;edge.push_back(outer*(u*std::cos(t)+v*std::sin(t)));}
     riskEdgeProbes[id]=edge.front();ballCenterProbes[id]=center;
     Ribbon("risk_edge_"+std::to_string(id),edge,center,C(.58,.43,.78,1),.38,.02);AttachBall(entries.at("risk_edge_"+std::to_string(id)).visual,id);
     }else{riskEdgeProbes[id]=V::Zero;ballCenterProbes[id]=center;}
     double speed=motion.velocity.Length();std::vector<V>arrow;
     if(speed>1e-9){
       V direction=motion.velocity/speed;double length=.7*speed;V start=p.radius*direction,end=start+.7*motion.velocity;
       V side=direction.Cross(Camera()-center);if(side.Length()<1e-8)side=direction.Cross(V::UnitZ);if(side.Length()<1e-8)side=direction.Cross(V::UnitY);side.Normalize();double head=std::min(.14,length*.4);
       arrow={start,end,end,end-head*direction+head*.45*side,end,end-head*direction-head*.45*side};
       minimumArrowAlignment=std::min(minimumArrowAlignment,(end-start).Normalized().Dot(direction));
     }
     Ribbon("speed_"+std::to_string(id),arrow,center,C(.91,.12,.045,1),1.,.04,false);
     auto&glyph=entries.at("speed_"+std::to_string(id));AttachBall(glyph.visual,id);maximumArrowAnchorError=std::max(maximumArrowAnchorError,(glyph.visual->WorldPose().Pos()-center).Length());
   }
 }



 void AlignPhysicsFrame(){
   // Match actual telemetry stamps. No state extrapolation and no physics writes.
   for(auto at=motionFrames.rbegin();at!=motionFrames.rend();++at){
     auto pose=truthFrames.lower_bound(at->first-1e-9);
     if(pose==truthFrames.end()||std::abs(pose->first-at->first)>1e-8)continue;
     worldTime=at->first;dronePose=pose->second;motions=at->second;haveTruth=true;
     for(const auto&[id,motion]:motions){
       actualObstacles[id]=motion.position;auto ball=Ball(id);
       if(ball)ball->SetWorldPosition(motion.position); // rendering visual only
     }
     break;
   }
 }

 void PublishFrameAudit(){
   if(time<0||!haveTruth||parameters.empty()||phase==2||(phase!=0&&terminalFrameAudited))return;if(phase!=0)terminalFrameAudited=true;
   gz::msgs::Double_V msg;V origin=dronePose.Pos();
   for(double v:std::vector<double>{worldTime,time.load(),double(++renderFrames),origin.X(),origin.Y(),origin.Z(),double(parameters.size()),maximumFormulaError,maximumAttachmentError,maximumArrowAnchorError,minimumArrowAlignment,double(phase)})msg.add_data(v);
   for(const auto&[id,p]:parameters){
     for(double v:std::vector<double>{double(id),p.e.X(),p.e.Y(),p.e.Z(),p.vertex,p.lambda,p.R,p.Vr,p.ref,p.stamp,p.radius,p.base,p.span,p.cap,p.scale,p.Tr,p.beta,p.clearance})msg.add_data(v);
     auto found=meshProbes.find(id);
     for(int j=0;j<3;j++){V v=(found!=meshProbes.end()&&found->second.size()==3)?found->second[j]:V::Zero;msg.add_data(v.X());msg.add_data(v.Y());msg.add_data(v.Z());}
     for(auto v:std::vector<V>{riskEdgeProbes[id],ballCenterProbes[id]}){msg.add_data(v.X());msg.add_data(v.Y());msg.add_data(v.Z());}
   }
   if(auditFile.is_open()){
     std::uint32_t size=msg.data_size();auditFile.write(reinterpret_cast<const char*>(&size),4);
     for(double value:msg.data())auditFile.write(reinterpret_cast<const char*>(&value),8);
     auditFile.flush();
   }
   frameAuditPub.Publish(msg);
 }

 bool eventFilter(QObject*obj,QEvent*event)override{
   if(event->type()==gz::gui::events::PreRender::kType){
     auto now=std::chrono::steady_clock::now();
     if(nextDisplayFrame>now)std::this_thread::sleep_until(nextDisplayFrame);
     nextDisplayFrame=std::chrono::steady_clock::now()+std::chrono::milliseconds(50);
     if(!scene)Init();if(scene){
       std::lock_guard<std::mutex>lock(mutex);auto begin=std::chrono::steady_clock::now();bool applied=dirty;AlignPhysicsFrame();Drone();
       if(applied){phase=0;for(auto&d:latest.header().data())if(d.key()=="prepare")phase=2;for(auto&m:latest.marker())Apply(m);time=latest.header().stamp().sec()+1e-9*latest.header().stamp().nsec();for(auto&d:latest.header().data())if(d.key()=="completed")phase=1;dirty=false;frames++;emit StatusChanged();}
       if(completionPending){phase=completionOutcome;completionPending=false;emit StatusChanged();}
       V currentCamera=Camera();
       bool changed=applied||!geometryCached||worldTime!=lastGeometryTime||currentCamera.X()!=lastGeometryCamera.X()||currentCamera.Y()!=lastGeometryCamera.Y()||currentCamera.Z()!=lastGeometryCamera.Z();
       if(changed){Boundaries(applied);HalosAndVelocity();lastGeometryTime=worldTime;lastGeometryCamera=currentCamera;geometryCached=haveTruth;}
       PublishFrameAudit();
       if(applied){gz::msgs::Double_V status;for(double v:std::vector<double>{time.load(),double(frames),double(replicaBuilt),double(entries.size()),std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count(),double(parameters.size()),maximumFormulaError,maximumAttachmentError,maximumArrowAnchorError,minimumArrowAlignment,minimumRadius,maximumRadius,double(motions.size()),double(attachedBalls)})status.add_data(v);lastStatus=status;statusPub.Publish(status);}
     }
   }
   return QObject::eventFilter(obj,event);
 }

};
GZ_ADD_PLUGIN(RAView,gz::gui::Plugin)
#include "RAView.moc"
