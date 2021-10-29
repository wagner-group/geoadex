#include <CGAL/config.h>
#define CGAL_EIGEN3_ENABLED
#if defined(BOOST_GCC) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 4)
#include <iostream>
int main()
{
  std::cerr << "NOTICE: This test requires G++ >= 4.4, and will not be compiled." << std::endl;
}
#else
#include <CGAL/Epick_d.h>
#include <eigen3/Eigen/Core>
#include <CGAL/Delaunay_triangulation.h>
#include <CGAL/IO/Triangulation_off_ostream.h>
#include <CGAL/point_generators_d.h>
#include <CGAL/Timer.h>
#include <CGAL/algorithm.h>
#include <CGAL/Origin.h>

#include <vector>
#include <random>
#include <string>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <boost/algorithm/string.hpp>
#include <chrono>

// uncomment this if want generate random points
// #define RANDOM_PTS 

const double lower_bound = 0.0;
const double upper_bound = 1.0;
const double step_size = 0.01;
const double small_step_size = 0.001;
//TODO: Change the path_prefix before running the function!
const std::string path_prefix = "/Users/angelynaye/Desktop/Research/result/space-partition-adv";

/** Helper Functions Section  **/

//function to read in data from csv file 
std::vector<std::vector<double> > read_data(std::string file_name, int dim,
std::string delimeter = " ")
{
	std::ifstream file(file_name);
 
	std::vector<std::vector<double> > data_list;
 
	std::string line = "";
	// Iterate through each line split the content using delimeter
  // convert it to double
	while (getline(file, line))
	{
		std::vector<std::string> vec;
		boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
        std::vector<double> dd;
        dd.reserve(dim);
        for(std::vector<std::string>::iterator it = vec.begin(); 
        it != vec.end(); ++it) {
          dd.push_back(std::stod(*it));
        }

		data_list.push_back(dd);

	}
	// Close the File
	file.close();
 
	return data_list;
}

// return true if within [0,1]
template<int D>
bool check_boundary(
  typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d a) {
    for(int i = 0; i < D; i++) {
      if(a[i] < lower_bound || a[i] > upper_bound) {
        return false;
      }
    }
    return true;
}

// compute Euclidean distance of dimension D points 
template<int D>
double distance(
  typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d a, 
  typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d b) {
    double distance = 0;
    for(int i = 0; i < D; i++) {
      distance += pow(a[i] - b[i], 2);
    }
    return distance;
  }

// compute distance of all points in vector to another point v
template<int D>
std::vector<double> neighbors_distance(
  std::set<typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Vertex_handle> neighs, 
  typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d st_proj,
  typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d cur_n) {
    typedef CGAL::Epick_d<CGAL::Dimension_tag<D>> K;
    typedef CGAL::Delaunay_triangulation<K> DT;
    typedef typename DT::Point_d Point;
    typedef typename DT::Vertex_handle Vertex_handle;
    typedef std::set<Vertex_handle> Vertex_set;
    std::vector<double> nei_dists;
    for(typename Vertex_set::iterator neighs_it = neighs.begin(); 
    neighs_it != neighs.end(); neighs_it++) {
      Point n = (*neighs_it)->point();
      // remove the n on the voronoi edge from the neighbors list 
      if(n == cur_n) continue;
      nei_dists.push_back(distance<D>(n, st_proj));
    }
    return nei_dists;
  }

// return dot product of two points
template<int D>
double compute_dot_product(
  typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d a, 
  typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d b) {
    double product = 0;
    for(int i = 0; i < D; i++) {
      product += a[i] * b[i];
    }
    return product;
  }

// assign the bounding value for edge
template<int D>
typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d assign_vals(
  typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d a) {
    typedef CGAL::Epick_d<CGAL::Dimension_tag<D>> K;
    typedef CGAL::Delaunay_triangulation<K> DT;
    typedef typename DT::Point_d Point;
    double new_a[D];

    for(int i = 0; i < D; i++) {
      if(a[i] < lower_bound ) {
        new_a[i] = lower_bound;
      } else if (a[i] > upper_bound) {
        new_a[i] = upper_bound;
      }
    }
    Point result(&new_a[0], &new_a[D]);
    return result;
}

// return addition of point
template<int D>
typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d point_addition(
  typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d a, 
  typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d b) {
    typedef CGAL::Epick_d<CGAL::Dimension_tag<D>> K;
    typedef CGAL::Delaunay_triangulation<K> DT;
    typedef typename DT::Point_d Point;

    // std::vector<double> sum;
    double sum[D];
    // sum.reserve(D);
    for(int i = 0; i < D; i++) {
      sum[i] = a[i] + b[i];
      // sum.push_back(a[i] + b[i]);
    }
    Point result(&sum[0], &sum[D]);
    // Point result(&sum.at(0), &sum.at(sum.size() - 1));
    return result;
  }

// return multiplication of a point and a constant double
template<int D>
typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d point_mul(
  typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d a, 
  double b) {
    typedef CGAL::Epick_d<CGAL::Dimension_tag<D>> K;
    typedef CGAL::Delaunay_triangulation<K> DT;
    typedef typename DT::Point_d Point;

    // std::vector<double> mul;
    // mul.reserve(D);
    double mul[D];
    for(int i = 0; i < D; i++) {
      mul[i] = a[i]*b;
      // mul.push_back(a[i]*b);
    }
    Point result(&mul[0], &mul[D]);
    // Point result(&mul.at(0), &mul.at(mul.size() - 1));
    return result;
  }

// return whether right point is larger than left point in a given direction
template<int D>
bool is_larger(
  typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d right, 
  typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d left,
  typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d direction) {
    typedef CGAL::Epick_d<CGAL::Dimension_tag<D>> K;
    typedef CGAL::Delaunay_triangulation<K> DT;
    typedef typename DT::Point_d Point;

    Point diff = point_addition<D>(right, point_mul<D>(left,  -1.0));
    // since right = left + direction * constant
    // constant is the same for all direction, we only need to check first poistion's c
    double diff_0 = diff[0];
    double direction_0 = direction[0];
    double c = diff_0 / direction_0;
    return c > 0;
  }


// return the projection of a point on half space
template<int D>
typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d projection(
  typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d v, 
  typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d w,
  double c) {
    typedef CGAL::Epick_d<CGAL::Dimension_tag<D>> K;
    typedef CGAL::Delaunay_triangulation<K> DT;
    typedef typename DT::Point_d Point;

    // v - v*unit_w*w + c*unit_w 
    double w_norm = sqrt(compute_dot_product<D>(w,w));
    Point unit_w = point_mul<D>(w, 1/w_norm);
    double proj_normal_length = compute_dot_product<D>(v, unit_w);
    Point proj_normal = point_mul<D>(w, proj_normal_length);
    Point proj_v = point_addition<D>(v, proj_normal);
    Point offset = point_mul<D>(unit_w, c);
    return point_addition<D>(proj_v, offset);
  }

// binary search get start and end of the voronoi edge
template<int D>
typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d binary_search(
  std::set<typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Vertex_handle> neighs,
  typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d cur_n,
  typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d start, 
  typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d direction,
  typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d v,
  typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d w,
  double c,  double cur_dist, double min_nei_dist, bool reverse) {
    typedef CGAL::Epick_d<CGAL::Dimension_tag<D>> K;
    typedef CGAL::Delaunay_triangulation<K> DT;
    typedef typename DT::Point_d Point;

    Point original_dir = direction;
    Point left = start;
    Point right = start;
    Point right_proj;
    double mul = 2.0;
    std::vector<double> nei_dists;
    
    // reverse = true if we want find end point in case 1/ start/end point in case 2: 
    // first point closer to other neighs than v: distance(v, proj_pt) >= min_nei_dist
    if(reverse) {
      cur_dist *= -1.0;
      min_nei_dist *= -1.0;
    }

    // find the range of the point

    // find start in case 1: first point closer to v than to other neighs 
    // i.e. distance(v, proj_pt) <= min_nei_dist if reverse -> ">="
    while(check_boundary<D>(right) && cur_dist > min_nei_dist) {
      left = right;
      direction = point_mul<D>(direction, mul);
      right = point_addition<D>(right, direction);
      right_proj = projection<D>(right, w, c);
      cur_dist = reverse? -1.0 * distance<D>(right_proj, v) : distance<D>(right_proj, v);
      nei_dists = neighbors_distance<D>(neighs, right_proj, cur_n);
      min_nei_dist = *std::min_element(nei_dists.begin(), nei_dists.end());
      min_nei_dist = reverse? -1.0 * min_nei_dist : min_nei_dist;
    }

    // get `right` inside the boundary if it's out 
    // may because the step size is too large that we skip the voronoi edge
    if(!check_boundary<D>(right)) {
      if(cur_dist > min_nei_dist) {
        // case 1: cur_dist > min_nei_dist: redo it with small step size, 
        right = start;
        direction = original_dir;
        while(check_boundary<D>(right) && cur_dist > min_nei_dist) {
          left = right;
          right = point_addition<D>(right, direction);
          right_proj = projection<D>(right, w, c);
          cur_dist = reverse? -1.0 * distance<D>(right_proj, v) : distance<D>(right_proj, v);
          nei_dists = neighbors_distance<D>(neighs, right_proj, cur_n);
          min_nei_dist = *std::min_element(nei_dists.begin(), nei_dists.end());
          min_nei_dist = reverse? -1.0 * min_nei_dist : min_nei_dist;
        }
      } else {
        // case 2: cur_dist <= min_nei_dist: get all the points within bound
        right = assign_vals<D>(right);
      }
    }
    
    Point mid;
    Point mid_proj;
    // (right - left)/ direction > 0
    while(is_larger<D>(right, left, original_dir)) {
      Point diff = point_mul<D>(point_addition<D>(right, point_mul<D>(left, -1.0)), 1/2.0);
      // get mid point
      mid = point_addition<D>(left, diff);
      mid_proj = projection<D>(mid, w, c);
      cur_dist = reverse? -1.0 * distance<D>(mid_proj, v) : distance<D>(mid_proj, v);
      nei_dists = neighbors_distance<D>(neighs, mid_proj, cur_n);
      min_nei_dist = *std::min_element(nei_dists.begin(), nei_dists.end());
      min_nei_dist = reverse? -1.0 * min_nei_dist : min_nei_dist;

      // if midpoint out of bound or midpoint is closer to v than all the other neighbors
      if(!check_boundary<D>(mid) || cur_dist <= min_nei_dist) {
        right = mid;
      } else {
        left = point_addition<D>(mid, original_dir);
      }
    }

    return right;
  }

template<int D>
std::vector<typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d>
find_voronoi_edge(
std::set<typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Vertex_handle> neighs,
typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d cur_n,
typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d lp,
typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d v, 
typename CGAL::Delaunay_triangulation<CGAL::Epick_d<CGAL::Dimension_tag<D>>>::Point_d w, double c) {
  
  typedef CGAL::Epick_d<CGAL::Dimension_tag<D>> K;
  typedef CGAL::Delaunay_triangulation<K> DT;
  typedef typename DT::Point_d Point;
  typedef typename DT::Vertex_handle Vertex_handle;
  typedef std::set<Vertex_handle> Vertex_set;

  std::vector<double> dummy = std::vector<double>(D + 1, 0.5);
  int end = dummy.size() - 1;

  Point start(&dummy.at(0), &dummy.at(end));

  double sum = 0;
  // projection of point onto plane
  Point st_proj = projection<D>(start, w, c);
  // calculate v's neighbors' distance 
  std::vector<double> nei_dists = neighbors_distance<D>(neighs, st_proj, cur_n);

  double cur_dist = distance<D>(st_proj, v);
  double min_nei_dist = *std::min_element(nei_dists.begin(), nei_dists.end());
  
  Point start_vor;
  Point end_vor;

  // generate random direction to walk [current random seed based on time]
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator (seed);
  std::uniform_int_distribution<int> dist(-1 , 1);
  double direct = dist(generator); // -1 or 1 
  direct *= step_size;

  // case 1: st_proj isn’t included in the voronoi edge 
  if(cur_dist > min_nei_dist) {
    Point point_direct = point_mul<D>(lp, direct);
    Point st2 = point_addition<D>(start, point_direct);
    Point st2_proj = projection<D>(st2, w, c);
    double temp_dist = distance<D>(st2_proj, v);
    //oof, we are moving the wrong direction:
    if(temp_dist > cur_dist) {
      point_direct = point_mul<D>(point_direct, -1.0); // flip the direction
    }
    
    start_vor = 
    binary_search<D>(neighs, cur_n, start, point_direct, v, w, c, cur_dist, min_nei_dist, false);

    Point start_vor_proj = projection<D>(start, w, c);
    double dist = distance<D>(start_vor_proj, v);
    nei_dists = neighbors_distance<D>(neighs, start_vor_proj, cur_n);
    double temp_min_nei_dist = *std::min_element(nei_dists.begin(), nei_dists.end());
    end_vor = 
    binary_search<D>(neighs, cur_n, start_vor, point_direct, v, w, c, dist, temp_min_nei_dist, true);

  } else {
    Point point_direct = point_mul<D>(w, direct);
    Point another_point_direct = point_mul<D>(point_direct, -1.0); // flip the direction
    start_vor = 
    binary_search<D>(neighs, cur_n, start, point_direct, v, w, c, cur_dist, min_nei_dist, true);
    end_vor =  
    binary_search<D>(neighs, cur_n, start, another_point_direct, v, w, c, cur_dist, min_nei_dist, true);
  }

  std::vector<Point> result;
  result.reserve(2);
  result.push_back(start_vor);
  result.push_back(end_vor);

  return result;
}

// build LSH
template<int D>
std::vector<std::vector<std::vector<double>>> compute_LSH(std::string file_name, 
std::size_t N, std::size_t num_proj, double bucket_size)
{
  typedef CGAL::Epick_d<CGAL::Dimension_tag<D>> K;
  typedef CGAL::Delaunay_triangulation<K> DT;

  typedef typename DT::Vertex Vertex;
  typedef typename DT::Vertex_handle Vertex_handle;
  typedef typename DT::Full_cell Full_cell;
  typedef typename DT::Full_cell_handle Full_cell_handle;
  typedef typename DT::Facet Facet;
  typedef typename DT::Point_d Point;
  typedef typename DT::Geom_traits::RT RT;
  typedef typename DT::Finite_full_cell_const_iterator Finite_full_cell_const_iterator;
  typedef typename DT::Finite_vertex_iterator Finite_vertex_iterator;
  typedef typename DT::Vertex_iterator Vertex_iterator;
  typedef typename DT::Face Face;
  typedef std::set<Vertex_handle> Vertex_set;
  typedef std::vector<Face> Faces;
  typedef CGAL::Random_points_in_cube_d<Point> Random_points_iterator;

  CGAL::Timer cost;  // timer
  std::vector<Point> points;

  // Generate points
  #ifdef RANDOM_PTS
  CGAL::Random rng;
  Random_points_iterator rand_it(D, 1.0, rng); // generate point within the cube with length 1
  std::copy_n(rand_it, N, std::back_inserter(points));
  #endif

  std::string full_file_name = path_prefix + file_name;

  // CSVReader reader(full_file_name);
  std::cout <<"          Reading file: " << file_name << std::endl;

  // Get the data from CSV File
  std::vector<std::vector<double> > data_vec = read_data(full_file_name, D);

  // construct points
  for(int i = 0; i < data_vec.size(); i++) {
    std::vector<double> cur = data_vec.at(i);
    double temp[cur.size()];
    std::copy(cur.begin(), cur.end(), temp);
    Point p(&temp[0], &temp[cur.size()]);
    points.push_back(p);
  }

  #ifdef READ_PTS
  for(int i = 0; i < D; i++) {
    std::cout << "Points 0[ " << i << "]"<< points.at(0)[i] << std::endl;
  }
  #endif

  cost.reset();
  cost.start();

  N = data_vec.size();
  std::cout << "Delaunay triangulation of " << N <<
    " points in dim " << D << ":" << std::endl;

  // create delaunay triangulation with dimension D
  DT dt(D);
  
  dt.insert(points.begin(), points.end());

  // assert the delaunay triangle is valid
  CGAL_assertion(dt.is_valid());

  // generate random line passing through (0,0), stored it in vector<Point> w
  std::default_random_engine generator;
  double cur[D] = {};
  // randomly generate pts from standard normal distribution
  std::normal_distribution<double> distribution(0.0, 1.0);
  std::vector<Point> proj_lines;

  double sum_sqr;
  for(int i= 0; i< num_proj; ++i) {
    sum_sqr = 0.0;
    for(int j = 0; j < D; j++) {
        cur[j]= distribution(generator);
        sum_sqr += pow(cur[j], 2);
    }

    // check not divide by zero, add a really small constant 
    // divide by norm + e^-10
    double norm = sqrt(sum_sqr);
    if(norm == 0) norm += 1e-9;
    for(int j = 0; j < D; j++) {
        cur[j] /= norm;
    }

    Point p(&cur[0], &cur[D]);
    proj_lines.push_back(p);
    std::cout << "\n" << std::endl;
  }

  int max_bucket = 2 * std::ceil(sqrt(D) / bucket_size); 
  int bound = max_bucket / 2;
  int num_edges = 0;

  // generate all the neighs for each vertices 
  // calculate the #edges for all 
  Vertex_iterator fvit = dt.vertices_begin();
  for (;fvit != dt.vertices_end(); fvit++) {
    if(dt.is_infinite(fvit)) continue;
    Vertex_handle curr = fvit;
    // circulate through incident full cells to get all neighbors of current vertex
    std::vector<Full_cell_handle> neigh_full_cellss;
    dt.tds().incident_full_cells(curr, back_inserter(neigh_full_cellss));
    Vertex_handle vhh;
    for(typename std::vector<Full_cell_handle>::iterator it = neigh_full_cellss.begin(); 
    it != neigh_full_cellss.end(); ++it ) {
        for( int i = 0; i <= dt.current_dimension(); ++i )
        {
            vhh = (*it)->vertex(i);
            if( dt.is_infinite(vhh) || vhh ==  curr)
                continue;
            num_edges++;
        }
    }
  }


  std::vector<std::vector<std::vector<double>>> buks(num_proj, 
  std::vector<std::vector<double>>(max_bucket, std::vector<double>(num_edges)));

  // double buks[num_proj][max_bucket][num_edges];

  // iterate through all vertices
  fvit = dt.vertices_begin();
  for (;fvit != dt.vertices_end(); fvit++) {
    if(dt.is_infinite(fvit)) continue;
    Vertex_handle cur = fvit;
    Point v = cur->point();
    // circulate through incident full cells to get all neighbors of current vertex
    std::vector<Full_cell_handle> neigh_full_cells;
    dt.tds().incident_full_cells(cur, back_inserter(neigh_full_cells));
    Vertex_set neighs;
    Vertex_handle vh;
    for(typename std::vector<Full_cell_handle>::iterator it = neigh_full_cells.begin(); 
    it != neigh_full_cells.end(); ++it ) {
        for( int i = 0; i <= dt.current_dimension(); ++i )
        {
            vh = (*it)->vertex(i);
            if( dt.is_infinite(vh) || vh ==  cur)
                continue;
            neighs.insert(vh);
        }
    }
    
    int idx = 0;
    Point n;
    // iterate through neighbors
    for(typename Vertex_set::iterator neighs_it = neighs.begin(); 
    neighs_it != neighs.end(); neighs_it++) {
      n = (*neighs_it)->point();

      // find the half space
      Point w = point_addition<D>(v, point_mul<D>(n, -1.0)); // v - n

      Point half_point = point_mul<D>(point_addition<D>(n, v), 1 /2.0); // (n + v) / 2
      
      double product = compute_dot_product<D>(w, half_point);

      // iterate through project line
      for(std::size_t i = 0; i != proj_lines.size(); i++) {
        Point lp = proj_lines[i];
        std::vector<Point> edge = find_voronoi_edge<D>(neighs, n, lp, v, w, product);

        Point start = edge.at(0);
        Point end = edge.at(1);
        // project the start and end point to the line
        double a = compute_dot_product<D>(w, start);
        double b = compute_dot_product<D>(w, end);
        int s_idx = std::max(ceil(std::min(a,b)), lower_bound);
        int e_idx = std::min(upper_bound, ceil(std::max(a,b)));

        std::cout<< "s_idx " << "line: "<<i<< " neigh: "<<idx << " is " << s_idx << std::endl;
        std::cout<< "e_idx " << "line: "<<i<< " neigh: "<<idx << " is " << e_idx << std::endl;

        for(int s = s_idx; s != e_idx + 1; s++) {
          // buks[i][s][idx] += 1;
          buks.at(i).at(s).at(idx) += 1;
        }
      }
      // std::cout << "I am here ------------" << std::endl;
      idx++;
    }
  }

  double timing = cost.time();

  std::cout<< "Total computation time is: " << timing << std::endl;
  return buks;

}


int main(int argc, char **argv)
{
    srand(static_cast<unsigned int>(time(NULL)));

    std::vector<std::vector<std::vector<double>>> lsh_buk = 
    compute_LSH<2>("data.csv", 10, 5, 0.1);

    return 0;
}
#endif

// FIXME: low dimension, 3 , put 5 - 10 points, know exactly the edges 
// fix the projection line [eliminate all randomness]
// get the neighbors, check correct
// get the voronoi edges, check
// test the projection
// 