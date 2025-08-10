import './style.css'
import * as THREE from 'three';
import * as sushi from 'milsushi2'
import * as dat from 'dat.gui';
import Stats from 'three/examples/jsm/libs/stats.module.js'; 
import experiment_data from './demo_dataset.json' assert {type: 'json'};

// Define Variables
var camera, scene, renderer;
let stats;
var xyz_left, xyz_right, xyz_rightR, R, sm, vsa, shrink_level;
var cam_dist;

// Define fov & rotations to account for path along offset
var aspect = window.innerWidth / window.innerHeight;
const frustumSize = 2;
const left_x = -1;
const right_x = 1;
const ppi = Math.sqrt(2160**2 + 3840**2) / 32
const real_depth = 20;
var fov = 180 / Math.PI * 2 * Math.atan(3840/(2 * aspect * real_depth * ppi))
// console.log("FOV = ", fov)

// Define Renderer
renderer = new THREE.WebGLRenderer({antialias:true})
renderer.setSize( window.innerWidth, window.innerHeight );
renderer.setPixelRatio(devicePixelRatio)

// Load initial slant angles 
var sa = []
for (var i=0, len=experiment_data.xyz.length; i<len; i++) { 
  sa.push(Math.PI / 180 * (20 + 50 * Math.random()))
}
var sa_init = [...sa]

// Start
var k = 0;
init()
updateObject(k)
animate();


function rotx(theta){
  var tr = theta * Math.PI / 180
  var m = [ [         1        ,       0      ,  0], 
          [        0        ,  Math.cos(tr), -1 * Math.sin(tr)], 
          [        0        ,  Math.sin(tr), Math.cos(tr)] ];
  m = sushi.jsa2mat(m)
  return m
}

function roty(theta){
  var tr = theta * Math.PI / 180
  var m = [ [Math.cos(tr)      , 0, Math.sin(tr)], 
            [   0             , 1,     0       ], 
            [-1 * Math.sin(tr), 0, Math.cos(tr)] ];
  m = sushi.jsa2mat(m)
  return m
}


function scale_mat(theta_current, theta_target, shrink){
  var m = [ [         shrink        ,       0      ,  0], 
            [        0        ,  shrink, 0], 
            [shrink * (Math.cos(2*theta_current) - Math.cos(2*theta_target))/Math.sin(2*theta_target), 0,  shrink*Math.sin(2*theta_current)/Math.sin(2*theta_target)] ];
  m = sushi.jsa2mat(m)
  return m
}

function opf(theta){
  var object = scene.getObjectByName( "right" );
  sm = scale_mat(vsa*Math.PI/180, theta, shrink_level)
  xyz_rightR = sushi.mtimes(sm, xyz_right)
  object.geometry.setAttribute( 'position', new THREE.BufferAttribute( xyz_rightR._data, 3 ) );
  object.geometry.computeVertexNormals()
}


function onDocumentKeyDown(event) {
    var keyCode = event.which;
    if (keyCode == 0x57){
      // w increases slant by .01
      if (sa[k] < Math.PI/2-.01){
      sa[k] += .01
      opf(sa[k])
      }
    } else if (keyCode == 0x53){
      // s decreases slant angle by .01
      if (sa[k] > .01){
        sa[k] -= .01
        opf(sa[k])
      }
    } else if (keyCode == 0x20){
      // Space bar to go to next trial
      if (k < experiment_data.xyz.length-1){
        k += 1
        init(k)
        animate();
      }
    } else if (keyCode == 0x50){
      //p to go to previous trial
      if (k > 0){
        k -= 1
        init(k)
        animate();
      }
    }
}

function init(){ 

  // Make canvas
  scene = new THREE.Scene() 

  // Update Title
  document.getElementById("info").innerHTML = "Trial 1 / " + (experiment_data.xyz.length).toString();

  // Define Camera 
  const aspect = window.innerWidth / window.innerHeight;
  camera = new THREE.PerspectiveCamera(fov, aspect, .1, 50) 
  // camera = new THREE.OrthographicCamera(frustumSize * aspect / - 2, frustumSize * aspect / 2, frustumSize / 2, frustumSize / - 2, 1, 1000 );
  cam_dist = 4;
  camera.position.z = cam_dist;
  const theta_correction_left = Math.PI/2 - Math.atan(cam_dist/(Math.abs(left_x) + 1e-8))
  const theta_correction_right = Math.PI/2 - Math.atan(cam_dist/(right_x))
  // console.log("theta_correction = ", theta_correction_left)
  const preRight = roty(-1 * 180 / Math.PI * theta_correction_right)
  document.body.appendChild( renderer.domElement );

  // Lights 
  const light = new THREE.DirectionalLight(0xffffff, 6)
  light.position.set(-10,10,10)
  scene.add( light );

  // const light2 = new THREE.DirectionalLight(0xffffff, 1)
  // light2.position.set(10,10,10)
  // scene.add( light2 );

  const lightAmbient = new THREE.AmbientLight( 0xffffff ); //yellow = 0xf4fc03
  scene.add(lightAmbient)

  // Material 
  const material = new THREE.MeshPhongMaterial({color: 0xA9A9A9}); //darker grey A9A9A9, light grey D3D3D3
  material.side = THREE.DoubleSide;
  material.wireframe = false;
  material.roughness = 0;
  material.flatShading=true;


  // Define Left Object
  xyz_left = sushi.t(sushi.jsa2mat(experiment_data.xyz[k]))
  var preLeft = roty(180 / Math.PI * theta_correction_left)
  xyz_left = sushi.mtimes(preLeft, xyz_left)
  var geometry_left = new THREE.BufferGeometry();
  geometry_left.setAttribute( 'position', new THREE.Float32BufferAttribute(xyz_left._data, 3 ) );
  geometry_left.setIndex( experiment_data.triples );
  geometry_left.computeVertexNormals()
  geometry_left.name = "left"
  const left_cube = new THREE.Mesh(geometry_left, material);
  scene.add(left_cube);
  left_cube.position.x = left_x;
  // left_cube.rotation.x = experiment_data.tx[experiment_data.order[k]] * Math.PI / 180
  // left_cube.rotation.y = experiment_data.ty[experiment_data.order[k]] * Math.PI / 180
  // left_cube.rotation.z = experiment_data.tz[experiment_data.order[k]] * Math.PI / 180
  left_cube.name = "left"
  var edge_material = new THREE.LineBasicMaterial( { color: 0x000000 } );
  var wire_left = new THREE.LineSegments( geometry_left, edge_material );
  wire_left.position.x = left_x;
  scene.add( wire_left );


  // Define Right Object
  xyz_right = sushi.t(sushi.jsa2mat(experiment_data.xyz[k]))
  R = sushi.jsa2mat(experiment_data.R[k])
  xyz_right = sushi.mtimes(R, xyz_right)
  vsa = experiment_data.vsa[k]
  shrink_level = .5 + .2 * Math.random()
  sm = scale_mat(vsa*Math.PI/180, sa[k], shrink_level)
  xyz_rightR = sushi.mtimes(sm, xyz_right)
  var geometry_right = new THREE.BufferGeometry();
  geometry_right.setAttribute( 'position', new THREE.Float32BufferAttribute( xyz_rightR._data, 3 ) );
  geometry_right.setIndex( experiment_data.triples );
  geometry_right.computeVertexNormals()
  geometry_right.name = "right"
  const right_cube = new THREE.Mesh(geometry_right, material);
  right_cube.name = "right"
  scene.add(right_cube);
  right_cube.position.x = right_x;
  right_cube.rotation.y = Math.PI/2
  var edge_material = new THREE.LineBasicMaterial( { color: 0x000000 } );
  var wire_left = new THREE.LineSegments( geometry_left, edge_material );
  wire_left.position.x = left_x;
  scene.add( wire_left );
  var wire_right = new THREE.LineSegments( geometry_right, edge_material );
  wire_right.position.x = right_x;
  wire_right.rotation.y = Math.PI/2
  wire_right.name = "right_wire"
  scene.add( wire_right );

  // Add Event Listeners
  document.addEventListener("keydown", onDocumentKeyDown, false);
  window.addEventListener( 'resize', onWindowResize );
  document.getElementById("db").addEventListener("click", onDownload, false);

  // Stats
  stats = Stats();
  document.body.appendChild( stats.dom );
}

function updateObject(k){
  
}


function onWindowResize() {
  var aspect = window.innerWidth / window.innerHeight;
  camera.aspect = aspect
  camera.updateProjectionMatrix();
  renderer.setSize( window.innerWidth, window.innerHeight );
}


function animate() {
  requestAnimationFrame( animate );
  var object = scene.getObjectByName( "right" );
  var wireframe = scene.getObjectByName( "right_wire" );
  const rot_speed = performance.now() * 0.0002;
  object.rotation.x = rot_speed * 4;
  wireframe.rotation.x = rot_speed * 4;
  renderer.render( scene, camera );
  // console.log(performance || Date)
  stats.update()
}



function onDownload(){
  // console.log(sa)
  var a = window.document.createElement('a');
  const user_data = ['trial,path,veridical_sa,recovered_sa,sa_init,subject,dist\n']
  var t, p, isa, vsa, rsa, s, j, set_label, d
  for (var i=0, len=100; i<len; i++){
    j = n_trials_per_set * set
    console.log("i,j = ", i,j)
    t = (i+j+1).toString()
    p = experiment_data['order'][i+j]
    vsa = Math.abs(experiment_data['vsa'][p])
    isa = sa_init[i+j] * 180 / Math.PI
    rsa = (180/Math.PI * sa[i+j]).toString()
    d = experiment_data['dists'][i+j]
    user_data.push(t+","+p+","+vsa+","+rsa+","+isa+","+localStorage.getItem("initials").trim()+","+d.toString()+"\n")
  }
  a.href = window.URL.createObjectURL(new Blob(user_data, {type: 'text/csv'}));
  set_label = (parseInt(localStorage.getItem("set")) + 1).toString()
  a.download = 'condition3_' + localStorage.getItem("initials") +'_follow_up'  + '.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}






