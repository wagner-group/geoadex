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
#include <CGAL/Memory_sizer.h>

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <iterator>
#include <algorithm>
#include <unistd.h>
#include <boost/algorithm/string.hpp>
#define OUTPUT_STATS
 
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


void test(int dim, std::string file_name, std::string output_file)
{
    typedef CGAL::Epick_d<CGAL::Dynamic_dimension_tag> K;
    typedef CGAL::Delaunay_triangulation<K> DT;
    typedef typename DT::Point Point;
    typedef CGAL::Random_points_in_cube_d<Point> Random_points_iterator;
    CGAL::Timer timer;  // timer

    //TODO: Change the path_prefix before running the function!
    std::string path_prefix = "/Users/angelynaye/desktop/research/data/";
    std::string full_file_name = path_prefix + file_name;

    std::vector<Point> points;

    // CSVReader reader(full_file_name);
    std::cout <<"          Reading file: " << file_name << std::endl;
    // Get the data from CSV File
    std::vector<std::vector<double> > data_vec = read_data(full_file_name, dim);



    std::size_t N  = data_vec.size();

    std::cout << N <<std::endl;
    int i = 0;

    // construct points
    for(int i = 0; i < data_vec.size(); i++) {
      std::vector<double> cur = data_vec.at(i);
      double temp[cur.size()];
      std::copy(cur.begin(), cur.end(), temp);
      Point p(&temp[0], &temp[cur.size()]);
      points.push_back(p);
    }

    std::size_t mem_before = CGAL::Memory_sizer().virtual_size();
    timer.reset();
    timer.start();

    // Build the Regular Triangulation
    DT dt(dim);


    // std::istream_iterator<Point> begin (iFile), end;
    dt.insert(points.begin(), points.end());

  
    std::cout << "Delaunay triangulation of " << N <<
    " points in dim " << dim << ":" << std::endl;

    std::size_t mem = CGAL::Memory_sizer().virtual_size() - mem_before;
    double timing = timer.time();
    std::cout << "  Triangles Complete in " << timing << " seconds." << std::endl;
    std::cout << "  Memory consumption: " << (mem >> 10) << " KB.\n";
    std::size_t nbfc= dt.number_of_finite_full_cells();
    std::size_t nbc= dt.number_of_full_cells();
    std::cout << "There are " << dt.number_of_vertices() << " vertices, " 
            << nbfc << " finite simplices and " 
            << (nbc-nbfc) << " convex hull Facets.\n"
            << std::endl;

    #ifdef OUTPUT_STATS
    path_prefix += "stats/";
    std::string output_name = path_prefix + output_file;
    std::ofstream csv_file(output_name);
    csv_file 
        << "Dimension: " << dim << "; "
        << "Numbers of Pts: " << N << "; "
        << "Completion Time: "<< timing << " seconds; "
        << "Used Memory: " << mem << " KB; "
        << "Numbers of Facets: "<< nbfc << "\n "
        << std::flush;
    #endif

}

int main()
{
  int dims[13] = { 
    // 2, 8, 
    2, 10, 25, 25, 25, 14, 22, 22, 24, 54, 54, 60, 7}; 
  std::string names[13] = {
      // "fourclass.csv",
      // "diabetes.csv",
      "halfmoon.csv",
      "cancer.csv",
      "mnist17_test.csv",
      "f-mnist06_test.csv",
      "f-mnist35.csv",
      "australian.csv",
      "svmguide3.csv",
      "ijcnn1.csv",
      "german.csv",
      "covtype_bi.csv",
      "cov_dat.csv",
      "splice.csv",
      "abalone.csv",
  };
  std::string output[13] = {
      // "fourclass_stat.csv",
      // "diabetes_stat.csv",
      "halfmoon_stat.csv",
      "breast_cancer_stat.csv",
      "mnist17_test_stat.csv",
      "f-mnist06_test_stat.csv",
      "f-mnist35_stat.csv",
      "australian_stat.csv",
      "svmguide3_stat.csv",
      "ijcnn1_stat.csv",
      "german_stat.csv",
      "covtype_bi_stat.csv",
      "cov_dat_stat.csv",
      "splice_stat.csv",
      "abalone_stat.csv"
  };

  for(int i = 0; i < sizeof(names)/sizeof(names[0]); i++) {
    test(dims[i], names[i], output[i]);
  }

  return 0;
}
#endif