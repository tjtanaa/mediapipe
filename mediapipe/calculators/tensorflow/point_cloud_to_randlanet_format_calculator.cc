// // Copyright 2019 The MediaPipe Authors.
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //      http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.
// //
// // Converts vector<float> (or vector<vector<float>>) to 1D (or 2D) tf::Tensor.

// // #include "mediapipe/calculators/tensorflow/vector_float_to_tensor_calculator_options.pb.h"
// #include "mediapipe/framework/calculator_framework.h"
// #include "mediapipe/framework/port/ret_check.h"
// #include "mediapipe/framework/port/status.h"
// #include "tensorflow/core/framework/tensor.h"
// #include "tensorflow/core/framework/types.h"
// #include "mediapipe/calculators/tensorflow/tensorflow_nearest_neighbor/cc/kernels/knn_.h"
// #include <string>
// #include <iostream>
// #include <fstream>

// namespace mediapipe {



// namespace tf = ::tensorflow;

// const std::string InputTag[] = {"POINT_CLOUD"};
// const std::string OutputTag[] = {"NEIGHBOR_INDEX_0",
//                                 "NEIGHBOR_INDEX_1",
//                                 "NEIGHBOR_INDEX_2",
//                                 "NEIGHBOR_INDEX_3",
//                                 "NEIGHBOR_INDEX_4",
//                                 "SUBPOINTS_0",
//                                 "SUBPOINTS_1",
//                                 "SUBPOINTS_2",
//                                 "SUBPOINTS_3",
//                                 "SUBPOINTS_4",
//                                 "POOL_I_0",
//                                 "POOL_I_1",
//                                 "POOL_I_2",
//                                 "POOL_I_3",
//                                 "POOL_I_4",
//                                 "UP_I_0",
//                                 "UP_I_1",
//                                 "UP_I_2",
//                                 "UP_I_3",
//                                 "UP_I_4",
//                                 "BATCH_XYZ_0",
//                                 "BATCH_XYZ_1",
//                                 "BATCH_XYZ_2",
//                                 "BATCH_XYZ_3",
//                                 "BATCH_XYZ_4",
//                                 "BATCH_FEATURE"};
// // The calculator expects one input (a packet containing a vector<float> or
// // vector<vector<float>>) and generates one output (a packet containing a
// // tf::Tensor containing the same data). The output tensor will be either
// // 1D or 2D with dimensions corresponding to the input vector float.
// // It will hold DT_FLOAT values.
// //
// // Example config:
// // node {
// //   calculator: "PointCloudToRandlanetFormatCalculator"
//     // input_side_packet: "SESSION:tf_model"
//     // input_stream: "POINT_CLOUD:point_cloud_tensor"
//     // output_stream: "NEIGHBOR_INDEX_0:NEIGHBOR_INDEX_0_tensor"
//     // output_stream: "NEIGHBOR_INDEX_1:NEIGHBOR_INDEX_1_tensor"
//     // output_stream: "NEIGHBOR_INDEX_2:NEIGHBOR_INDEX_2_tensor"
//     // output_stream: "NEIGHBOR_INDEX_3:NEIGHBOR_INDEX_3_tensor"
//     // output_stream: "NEIGHBOR_INDEX_4:NEIGHBOR_INDEX_4_tensor"
//     // output_stream: "SUBPOINTS_0:SUBPOINTS_0_tensor"
//     // output_stream: "SUBPOINTS_1:SUBPOINTS_1_tensor"
//     // output_stream: "SUBPOINTS_2:SUBPOINTS_2_tensor"
//     // output_stream: "SUBPOINTS_3:SUBPOINTS_3_tensor"
//     // output_stream: "SUBPOINTS_4:SUBPOINTS_4_tensor"
//     // output_stream: "POOL_I_0:POOL_I_0_tensor"
//     // output_stream: "POOL_I_1:POOL_I_1_tensor"
//     // output_stream: "POOL_I_2:POOL_I_2_tensor"
//     // output_stream: "POOL_I_3:POOL_I_3_tensor"
//     // output_stream: "POOL_I_4:POOL_I_4_tensor"
//     // output_stream: "UP_I_0:UP_I_0_tensor"
//     // output_stream: "UP_I_1:UP_I_1_tensor"
//     // output_stream: "UP_I_2:UP_I_2_tensor"
//     // output_stream: "UP_I_3:UP_I_3_tensor"
//     // output_stream: "UP_I_4:UP_I_4_tensor"
//     // output_stream: "BATCH_XYZ_0:BATCH_XYZ_0_tensor"
//     // output_stream: "BATCH_XYZ_1:BATCH_XYZ_1_tensor"
//     // output_stream: "BATCH_XYZ_2:BATCH_XYZ_2_tensor"
//     // output_stream: "BATCH_XYZ_3:BATCH_XYZ_3_tensor"
//     // output_stream: "BATCH_XYZ_4:BATCH_XYZ_4_tensor"
//     // output_stream: "BATCH_FEATURE:BATCH_FEATURE_tensor"
// // }
// class PointCloudToRandlanetFormatCalculator : public CalculatorBase {
//  public:
//   static ::mediapipe::Status GetContract(CalculatorContract* cc);

//   ::mediapipe::Status Open(CalculatorContext* cc) override;
//   ::mediapipe::Status Process(CalculatorContext* cc) override;

// //  private:
// //   PointCloudToRandlanetFormatCalculatorOptions options_;
// };
// REGISTER_CALCULATOR(PointCloudToRandlanetFormatCalculator);

// ::mediapipe::Status PointCloudToRandlanetFormatCalculator::GetContract(
//     CalculatorContract* cc) {
// //   const auto& options = cc->Options<PointCloudToRandlanetFormatCalculatorOptions>();
//   // Start with only one input packet.
//   RET_CHECK_EQ(cc->Inputs().NumEntries(), 1)
//       << "Only one input stream is supported.";
// //   if (options.input_size() == 4) {
// //     cc->Inputs().Index(0).Set<tf::Tensor>(
// //         /* "Input vector<vector<float>>." */);
// //   }else {
// //     LOG(FATAL) << "input size not supported";
// //   }
  
//     cc->Inputs().Tag(InputTag[0]).Set<tf::Tensor>(
//         /* "Input vector<vector<float>>." */);
//   RET_CHECK_EQ(cc->Outputs().NumEntries(), 26)
//       << "Must have 27 output streams.";
//     for(int i = 0 ; i < 26;  i ++){
//         cc->Outputs().Tag(OutputTag[i]).Set<tf::Tensor>(
//             // Output stream with data as tf::Tensor and the same TimeSeriesHeader.
//         );        
//     }

//   return ::mediapipe::OkStatus();
// }

// ::mediapipe::Status PointCloudToRandlanetFormatCalculator::Open(CalculatorContext* cc) {
// //   options_ = cc->Options<PointCloudToRandlanetFormatCalculatorOptions>();
//   return ::mediapipe::OkStatus();
// }

// ::mediapipe::Status PointCloudToRandlanetFormatCalculator::Process(
//     CalculatorContext* cc) {



// // The input point cloud has been stored in a tensor object
//   const int init_batch_size = 1;
//   const int init_n_pts = 65536 ;//32768;
//   const int init_n_features = 3;
//   const int init_n_layers = 5;
//   const int K_cpp = 16; // hardcode parameter
//   const int sub_sampling_ratio[init_n_layers] = {4,4,4,4,2};


//   tf::TensorShape point_tensor_shape({init_batch_size, init_n_pts, init_n_features});

//   const tf::Tensor& point_tensor =
//       cc->Inputs().Tag(InputTag[0]).Value().Get<tf::Tensor>();

//   auto temp_point_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_FLOAT, point_tensor_shape);
//   for (int r = 0; r < init_n_pts ; ++r) {
//     for (int c = 0; c < init_n_features; ++c) {
//       temp_point_tensor->tensor<float, 3>()(0, r, c) = point_tensor.tensor<float, 3>()(0, r, c);
//       // if(r == 0){
//       //   std::cout << "r " << r <<  std::to_string(point_tensor.tensor<float, 3>()(0, r, c)) << std::endl;
//       // }
//     }
//   }

//   tf::TensorShape batch_feature_tensor_shape({init_batch_size, init_n_pts, init_n_features});
//   auto temp_batch_feature_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_FLOAT, batch_feature_tensor_shape);

//   std::string batch_feature_file = "/home/tan/tjtanaa/mediapipe/mediapipe/calculators/tensorflow/" + OutputTag[26-1] + ".txt";
//   std::ifstream file(batch_feature_file);
//   std::string line; 
//   std::string delimiter = " ";
//   int r = 0;
//   while (std::getline(file, line)) {
//     // process string ...
//     size_t pos = 0;
//     std::string token;
//     int count = 0;
//     // std::cout << line << std::endl;
//     // if(line == ""){
//     //   continue;
//     // }
//     while ((pos = line.find(delimiter)) != std::string::npos) {
//         token = line.substr(0, pos);
//         temp_batch_feature_tensor->tensor<float, 3>()(0, r, count) = std::stof(token) ;
//         // if (r < 5){
//         //   std::cout << token << "\t";
//         //   if (count == 2){
//         //     std::cout << std::endl;
//         //   }
//         // }

//         line.erase(0, pos + delimiter.length());
//         // std::cout << "count " << count << std::endl;
//         count++;
//     }
//     r++;
//     // std::cout << line << std::endl;
//   }

//     // std::cout << "temp_batch_feature_tensor.release()" << OutputTag[26-1] << std::endl;
//     cc->Outputs().Tag(OutputTag[26-1]).Add(temp_batch_feature_tensor.release(), cc->InputTimestamp());

//   int number_of_pts[] = {65536, 16384, 4096, 1024, 256,128};
// // =======
//   for(int layer = 0; layer < init_n_layers; layer++ ){
//     std::cout << "Layer: " << layer << std::endl;
//     // const int batch_size = temp_point_tensor->dim_size(0);
//     // const int npts = temp_point_tensor->dim_size(1);
//     // const int dim = temp_point_tensor->dim_size(2);
//     // const int nqueries = temp_point_tensor->dim_size(1);

//     const int batch_size = temp_point_tensor->dim_size(0);
//     const int npts = temp_point_tensor->dim_size(1);
//     const int dim = temp_point_tensor->dim_size(2);
//     const int nqueries = temp_point_tensor->dim_size(1);

//     std::cout << "layer " << layer  << "npts " << npts << "dim " << dim << "nqueries " << nqueries << std::endl;
//     std::cout << "npts/sub_sampling_ratio[layer] " << npts/sub_sampling_ratio[layer] << std::endl;
//     // create intermediate variables
//     tf::TensorShape neigh_idx_tensor_shape({batch_size, nqueries, K_cpp});
//     auto neigh_idx_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_INT64, neigh_idx_tensor_shape);
//     // auto neigh_idx_tensor_32 = ::absl::make_unique<tf::Tensor>(tf::DT_INT32, neigh_idx_tensor_shape);

//     tf::TensorShape sub_points_tensor_shape({batch_size, npts/sub_sampling_ratio[layer], dim});
//     auto sub_points_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_FLOAT, sub_points_tensor_shape);

//     tf::TensorShape pool_i_tensor_shape({batch_size, nqueries/sub_sampling_ratio[layer], K_cpp});
//     auto pool_i_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_INT64, pool_i_tensor_shape);
//     // auto pool_i_tensor_32 = ::absl::make_unique<tf::Tensor>(tf::DT_INT32, pool_i_tensor_shape);
    
//     tf::TensorShape up_i_tensor_shape({batch_size, npts, 1});
//     auto up_i_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_INT64, up_i_tensor_shape);
//     // auto up_i_tensor_32 = ::absl::make_unique<tf::Tensor>(tf::DT_INT32, up_i_tensor_shape);

//     std::cout << neigh_idx_tensor_shape << std::endl;
//     std::cout << sub_points_tensor_shape << std::endl;
//     std::cout << pool_i_tensor_shape << std::endl;
//     std::cout << up_i_tensor_shape << std::endl;


//     // Load Batch_XYZ
//     // ...
    
//     tf::TensorShape batch_xyz_tensor_shape({batch_size, number_of_pts[layer], dim});
//     auto batch_xyz_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_FLOAT, sub_points_tensor_shape);

//     std::string batch_xyz_file = "/home/tan/tjtanaa/mediapipe/mediapipe/calculators/tensorflow/" + OutputTag[20+layer] + ".txt";
//     std::cout << "batch_feature_file: " << batch_feature_file <<std::endl;
//     // std::ifstream ifile;
  
//     // ifile.open(batch_feature_file);
//     // if(ifile) {
//     //     std::cout<<"file exists"<< std::endl;
//     // } else {
//     //     std::cout<<"file doesn't exist" << std::endl;
//     // }

//     std::ifstream file1(batch_xyz_file);
//     std::string line = ""; 
//     std::string delimiter = " ";
//     int r = 0;
//     while (std::getline(file1, line)) {
//       // process string ...
//       size_t pos = 0;
//       std::string token;
//       int count = 0;
//       // std::cout << line << std::endl;
//       // if(line == ""){
//       //   continue;
//       // }
//       while ((pos = line.find(delimiter)) != std::string::npos) {
//           token = line.substr(0, pos);
//           // std::cout << token << "\n";
//             // try
//             // {
//             //   batch_xyz_tensor->tensor<float, 3>()(0, r, count) = std::stof(token) ;
//             // }
//             // catch (int e)
//             // {
//             //   std::cout << "An exception occurred. Exception Nr. " << e << '\n';
//             //   std::cout << "token: " << token << std::endl;
//             // }
//           batch_xyz_tensor->tensor<float, 3>()(0, r, count) = std::stof(token) ;
//           // if (r < 5){
//           //   std::cout << token << "\t";
//           //   if (count == 2){
//           //     std::cout << std::endl;
//           //   }
//           // }

//           line.erase(0, pos + delimiter.length());
//           // std::cout << "count " << count << std::endl;
//           count++;
//       }
//       r++;
//       // std::cout << line << std::endl;
//     }
//     cc->Outputs().Tag(OutputTag[20+layer]).Add(batch_xyz_tensor.release(), cc->InputTimestamp());
   
//     std::cout << "done" << std::endl;
//     // Load NEIGHBOR_INDEX
//     // ...

//     std::string neighbor_index_file = "/home/tan/tjtanaa/mediapipe/mediapipe/calculators/tensorflow/" + OutputTag[layer] + ".txt";
//     std::ifstream file2(neighbor_index_file);
//     // std::string line; 
//     // std::string delimiter = " ";
//     r = 0;
//     line = "";
//     while (std::getline(file2, line)) {
//       // process string ...
//       size_t pos = 0;
//       std::string token;
//       int count = 0;
//       // if(line == ""){
//       //   continue;
//       // }
//       while ((pos = line.find(delimiter)) != std::string::npos) {
//           token = line.substr(0, pos);
//           neigh_idx_tensor->tensor<long long int, 3>()(0, r, count) = std::stoll(token) ;
//           // if (r < 5){
//           //   std::cout << token << "\t";
//           //   if (count == 2){
//           //     std::cout << std::endl;
//           //   }
//           // }

//           line.erase(0, pos + delimiter.length());
//           // std::cout << "count " << count << std::endl;
//           count++;
//       }
//       r++;
//       // std::cout << line << std::endl;
//     }
//     cc->Outputs().Tag(OutputTag[layer]).Add(neigh_idx_tensor.release(), cc->InputTimestamp());
   
//     // Load POOL_I
//     // ...
//     std::string pool_file = "/home/tan/tjtanaa/mediapipe/mediapipe/calculators/tensorflow/" + OutputTag[10 +layer] + ".txt";
//     std::ifstream file3(pool_file);
//     // std::string line; 
//     // std::string delimiter = " ";
//     r = 0;
//     line = "";
//     while (std::getline(file3, line)) {
//       // process string ...
//       size_t pos = 0;
//       std::string token;
//       int count = 0;
//       // if(line == ""){
//       //   continue;
//       // }
//       while ((pos = line.find(delimiter)) != std::string::npos) {
//           token = line.substr(0, pos);
//           pool_i_tensor->tensor<long long int, 3>()(0, r, count) = std::stoll(token) ;
//           // if (r < 5){
//           //   std::cout << token << "\t";
//           //   if (count == 2){
//           //     std::cout << std::endl;
//           //   }
//           // }

//           line.erase(0, pos + delimiter.length());
//           // std::cout << "count " << count << std::endl;
//           count++;
//       }
//       r++;
//       // std::cout << line << std::endl;
//     }
//     cc->Outputs().Tag(OutputTag[10 +layer]).Add(pool_i_tensor.release(), cc->InputTimestamp());

//     // Load UP_I
//     // ...
//     std::string up_file = "/home/tan/tjtanaa/mediapipe/mediapipe/calculators/tensorflow/" + OutputTag[15 +layer] + ".txt";
//     std::ifstream file4(up_file);
//     // std::string line; 
//     // std::string delimiter = " ";
//     r = 0;
//     line = "";
//     while (std::getline(file4, line)) {
//       // process string ...
//       size_t pos = 0;
//       std::string token;
//       int count = 0;
//       // if(line == ""){
//       //   continue;
//       // }
//       while ((pos = line.find(delimiter)) != std::string::npos) {
//           token = line.substr(0, pos);
//           up_i_tensor->tensor<long long int, 3>()(0, r, count) = std::stoll(token) ;
//           // if (r < 5){
//           //   std::cout << token << "\t";
//           //   if (count == 2){
//           //     std::cout << std::endl;
//           //   }
//           // }

//           line.erase(0, pos + delimiter.length());
//           // std::cout << "count " << count << std::endl;
//           count++;
//       }
//       r++;
//       // std::cout << line << std::endl;
//     }
//     cc->Outputs().Tag(OutputTag[15 +layer]).Add(up_i_tensor.release(), cc->InputTimestamp());

//     // Load SUBPOINTS
//     // ...
//     std::string subpoints_file = "/home/tan/tjtanaa/mediapipe/mediapipe/calculators/tensorflow/" + OutputTag[5 +layer] + ".txt";
//     std::ifstream file5(subpoints_file);
//     // std::string line; 
//     // std::string delimiter = " ";
//     r = 0;
//     line = "";
//     while (std::getline(file5, line)) {
//       // process string ...
//       size_t pos = 0;
//       std::string token;
//       int count = 0;
//       // if(line == ""){
//       //   continue;
//       // }
//       while ((pos = line.find(delimiter)) != std::string::npos) {
//           token = line.substr(0, pos);
//           sub_points_tensor->tensor<float, 3>()(0, r, count) = std::stof(token) ;
//           // if (r < 5){
//           //   std::cout << token << "\t";
//           //   if (count == 2){
//           //     std::cout << std::endl;
//           //   }
//           // }

//           line.erase(0, pos + delimiter.length());
//           // std::cout << "count " << count << std::endl;
//           count++;
//       }
//       r++;
//       // std::cout << line << std::endl;
//     }
//     cc->Outputs().Tag(OutputTag[5 +layer]).Add(sub_points_tensor.release(), cc->InputTimestamp());


//     // // start to compute
//     // // auto pt_tensor = temp_point_tensor->flat<float>().data();
//     // // auto q_tensor = temp_point_tensor->flat<float>().data();
//     // auto pt_tensor = temp_point_tensor->flat<float>().data();
//     // auto q_tensor = temp_point_tensor->flat<float>().data();
//     // auto neigh_idx_flat = neigh_idx_tensor->flat<long long int>().data();
//     // cpp_knn_batch_omp(pt_tensor, batch_size, npts, dim, q_tensor, nqueries, K_cpp, neigh_idx_flat);
    
//     // for (int r = 0; r < npts/sub_sampling_ratio[layer] ; ++r) {
//     //   for (int c = 0; c < K_cpp; ++c) {
//     //     pool_i_tensor->tensor<long long int, 3>()(0, r, c) = neigh_idx_tensor->tensor<long long int, 3>()(0, r, c);
//     //     //   if(r ==0 & c == 0){
//     //     //     std::cout << "pool_i_tensor: " <<std::endl;
//     //     //   }
//     //     // // if(r == 0){
//     //     // //   std::cout << "pool_i_tensor: " << std::to_string(pool_i_tensor->tensor<long long int, 3>()(0, r, c)) << std::endl;
//     //     // // }

//     //     //   if (r < 5){
//     //     //     std::cout << std::to_string(pool_i_tensor->tensor<long long int, 3>()(0, c, r)) << "\t";
//     //     //   }
//     //     //   if(c == K_cpp - 1){
//     //     //     std::cout << std::endl;
//     //     // }
//     //   }
//     // }    

//     // // std::cout << "subpoint "  << npts/sub_sampling_ratio[layer] << std::endl;
//     // for (int r = 0; r < npts/sub_sampling_ratio[layer] ; ++r) {
//     //   for (int c = 0; c < dim; ++c) {
//     //     // sub_points_tensor.tensor<float, 3>()(0, r, c) = temp_point_tensor->tensor<float, 3>()(0, r, c);
//     //     sub_points_tensor->tensor<float, 3>()(0, r, c) = temp_point_tensor->tensor<float, 3>()(0, r, c);
//     //     // std::cout << "r " << r << " c " << c << std::endl;
//     //     // std::cout << sub_points_tensor.tensor<float, 3>()(0, r, c) << std::endl;
//     //   }
//     // }   

//     // std::cout << "temp_point_tensor.release()" << OutputTag[20+layer] << std::endl;
//     // cc->Outputs().Tag(OutputTag[20+layer]).Add(temp_point_tensor.release(), cc->InputTimestamp());

//     // // std::cout << "delete temp_point_tensor " << std::endl;
//     temp_point_tensor.release();
//     // // std::cout << "after deleting temp_point_tensor " << std::endl;
//     if(layer < 4){
//         temp_point_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_FLOAT, sub_points_tensor_shape);
//         for (int r = 0; r < npts/sub_sampling_ratio[layer] ; ++r) {
//             for (int c = 0; c < dim; ++c) {
//                 // temp_point_tensor->tensor<float, 3>()(0, r, c) = sub_points_tensor.tensor<float, 3>()(0, r, c);
//                 temp_point_tensor->tensor<float, 3>()(0, r, c) = sub_points_tensor->tensor<float, 3>()(0, r, c);
//                 // std::cout << sub_points_tensor.tensor<float, 3>()(0, r, c) << std::endl;
//             }
//         }
//     } 

//     // auto sub_points_flat = sub_points_tensor->flat<float>().data();

//     // auto up_i_flat = up_i_tensor->flat<long long int>().data();
//     // cpp_knn_batch_omp(sub_points_flat, batch_size, npts, dim, q_tensor, nqueries, 1, up_i_flat);

//     // for (int r = 0; r < 50 ; ++r) {
//     //   for (int c = 0; c < 1; ++c) {
//     //       // temp_point_tensor->tensor<float, 3>()(0, r, c) = sub_points_tensor.tensor<float, 3>()(0, r, c);
//     //       // up_i_tensor->tensor<int, 3>()(0, 0, c) = (int) up_i_tensor->tensor<long long int, 3>()(0, 0, c);
//     //       // std::cout << sub_points_tensor.tensor<float, 3>()(0, r, c) << std::endl;
//     //       if(r == 0 & c== 0){
//     //         std::cout << "up_i_tensor: " ;
//     //       }
          
//     //       std::cout <<  up_i_tensor->tensor<long long int, 3>()(0, r, c) << "\t";
//     //       if(r % 10 == 0){
//     //         std::cout << std::endl;
//     //       }
//     //   }
//     // }

//     // // std::string neigh_idx_tensor_name = "NEIGHBOR_INDEX_" + std::to_string(layer) + "_tensor";

//     // std::cout << "neigh_idx_tensor.release()" << OutputTag[layer] << std::endl;
//     // cc->Outputs().Tag(OutputTag[layer]).Add(neigh_idx_tensor.release(), cc->InputTimestamp());

//     // // std::string sub_points_tensor_name = "SUBPOINTS_" + std::to_string(layer) + "_tensor";

//     // std::cout << "sub_points_tensor.release()" << OutputTag[5 +layer]<< std::endl;
//     // cc->Outputs().Tag(OutputTag[5 +layer]).Add(sub_points_tensor.release(), cc->InputTimestamp());

//     // // std::string pool_i_tensor_name = "POOL_I_" + std::to_string(layer) + "_tensor";

//     // std::cout << "pool_i_tensor.release()" << OutputTag[10 +layer] << std::endl;
//     // cc->Outputs().Tag(OutputTag[10 +layer]).Add(pool_i_tensor.release(), cc->InputTimestamp());

//     // // std::string up_i_tensor_name = "UP_I_" + std::to_string(layer) + "_tensor";

//     // std::cout << "up_i_tensor.release()" << OutputTag[15 +layer] << std::endl;
//     // cc->Outputs().Tag(OutputTag[15 +layer]).Add(up_i_tensor.release(), cc->InputTimestamp());

//   }
//   std::cout << "return " <<std::endl;
//   return ::mediapipe::OkStatus();
// }

// }  // namespace mediapipe
















// ================= Original calculator =====================
// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Converts vector<float> (or vector<vector<float>>) to 1D (or 2D) tf::Tensor.

#include "mediapipe/calculators/tensorflow/point_cloud_to_randlanet_format_calculator_options.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "mediapipe/calculators/tensorflow/tensorflow_nearest_neighbor/cc/kernels/knn_.h"
#include <string>
#include <iostream>
namespace mediapipe {



namespace tf = ::tensorflow;

const std::string InputTag[] = {"POINT_CLOUD"};

const std::string OutputTag[] = {"NEIGHBOR_INDEX_0",
                                "NEIGHBOR_INDEX_1",
                                "NEIGHBOR_INDEX_2",
                                "NEIGHBOR_INDEX_3",
                                "NEIGHBOR_INDEX_4",
                                "POOL_I_0",
                                "POOL_I_1",
                                "POOL_I_2",
                                "POOL_I_3",
                                "POOL_I_4",
                                "UP_I_0",
                                "UP_I_1",
                                "UP_I_2",
                                "UP_I_3",
                                "UP_I_4",
                                "BATCH_XYZ_0",
                                "BATCH_XYZ_1",
                                "BATCH_XYZ_2",
                                "BATCH_XYZ_3",
                                "BATCH_XYZ_4",
                                "BATCH_FEATURE"};
// The calculator expects one input (a packet containing a vector<float> or
// vector<vector<float>>) and generates one output (a packet containing a
// tf::Tensor containing the same data). The output tensor will be either
// 1D or 2D with dimensions corresponding to the input vector float.
// It will hold DT_FLOAT values.
//
// Example config:
class PointCloudToRandlanetFormatCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  PointCloudToRandlanetFormatCalculatorOptions options_;
};
REGISTER_CALCULATOR(PointCloudToRandlanetFormatCalculator);

::mediapipe::Status PointCloudToRandlanetFormatCalculator::GetContract(
    CalculatorContract* cc) {
  // Start with only one input packet.
  RET_CHECK_EQ(cc->Inputs().NumEntries(), 1)
      << "Only one input stream is supported.";
    cc->Inputs().Tag(InputTag[0]).Set<tf::Tensor>(
        /* "Input vector<vector<float>>." */);
  RET_CHECK_EQ(cc->Outputs().NumEntries(), 21)
      << "Must have 21 output streams.";
    for(int i = 0 ; i < 21;  i ++){
      // std::cout << OutputTag[i] << std::endl;
      cc->Outputs().Tag(OutputTag[i]).Set<tf::Tensor>(
          // Output stream with data as tf::Tensor and the same TimeSeriesHeader.
      );        
    }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status PointCloudToRandlanetFormatCalculator::Open(CalculatorContext* cc) {
  options_ = cc->Options<PointCloudToRandlanetFormatCalculatorOptions>();
  return ::mediapipe::OkStatus();
}

::mediapipe::Status PointCloudToRandlanetFormatCalculator::Process(
    CalculatorContext* cc) {

std::cout << "PROCESS POINT CLOUD" << std::endl;

// The input point cloud has been stored in a tensor object
  const int init_batch_size = options_.batch_size();
  const int init_n_pts = options_.npts();
  const int init_n_features = options_.n_features();
  const int init_n_layers = options_.n_layers();
  const int K_cpp = options_.k_cpp(); // hardcode parameter
  const int sub_sampling_ratio[init_n_layers] = {4,4,4,4,2}; // hardcode parameter
  std::cout << "Options Parameters: " << std::to_string(init_batch_size) << "\t" <<
            std::to_string(init_n_pts) << "\t" <<
            std::to_string(init_n_features) << "\t" <<
            std::to_string(init_n_layers) << "\t" <<
            std::to_string(K_cpp) << "\t" << std::endl;

  tf::TensorShape point_tensor_shape({init_batch_size, init_n_pts, init_n_features});

  const tf::Tensor& point_tensor =
      cc->Inputs().Tag(InputTag[0]).Value().Get<tf::Tensor>();

  auto temp_point_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_FLOAT, point_tensor_shape);
  for (int r = 0; r < init_n_pts ; ++r) {
    for (int c = 0; c < init_n_features; ++c) {
      temp_point_tensor->tensor<float, 3>()(0, r, c) = point_tensor.tensor<float, 3>()(0, r, c);
      if(r == 0){
        std::cout << "r " << r <<  std::to_string(point_tensor.tensor<float, 3>()(0, r, c)) << std::endl;
      }
    }
  }

  tf::TensorShape batch_feature_tensor_shape({init_batch_size, init_n_pts, init_n_features});
  auto temp_batch_feature_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_FLOAT, point_tensor_shape);
  for (int r = 0; r < init_n_pts ; ++r) {
    for (int c = 0; c < init_n_features; ++c) {
      temp_batch_feature_tensor->tensor<float, 3>()(0, r, c) = point_tensor.tensor<float, 3>()(0, r, c);
      // temp_batch_feature_tensor->tensor<float, 3>()(0, r, c+3) = point_tensor.tensor<float, 3>()(0, r, c);
    }
  }    
//   std::string batch_feature_tensor_name = "BATCH_FEATURE_tensor";

    // std::cout << "temp_batch_feature_tensor.release()" << OutputTag[26-1] << std::endl;
    cc->Outputs().Tag(OutputTag[21-1]).Add(temp_batch_feature_tensor.release(), cc->InputTimestamp());
// =======
  for(int layer = 0; layer < init_n_layers; layer++ ){
    std::cout << "Layer: " << layer << std::endl;

    const int batch_size = temp_point_tensor->dim_size(0);
    const int npts = temp_point_tensor->dim_size(1);
    const int dim = temp_point_tensor->dim_size(2);
    const int nqueries = temp_point_tensor->dim_size(1);

    std::cout << "layer " << layer  << "npts " << npts << "dim " << dim << "nqueries " << nqueries << std::endl;
    std::cout << "npts/sub_sampling_ratio[layer] " << npts/sub_sampling_ratio[layer] << std::endl;
    // create intermediate variables
    tf::TensorShape neigh_idx_tensor_shape({batch_size, nqueries, K_cpp});
    auto neigh_idx_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_INT64, neigh_idx_tensor_shape);
    // auto neigh_idx_tensor_32 = ::absl::make_unique<tf::Tensor>(tf::DT_INT32, neigh_idx_tensor_shape);

    tf::TensorShape sub_points_tensor_shape({batch_size, npts/sub_sampling_ratio[layer], dim});
    auto sub_points_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_FLOAT, sub_points_tensor_shape);

    tf::TensorShape pool_i_tensor_shape({batch_size, nqueries/sub_sampling_ratio[layer], K_cpp});
    auto pool_i_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_INT64, pool_i_tensor_shape);
    // auto pool_i_tensor_32 = ::absl::make_unique<tf::Tensor>(tf::DT_INT32, pool_i_tensor_shape);
    
    tf::TensorShape up_i_tensor_shape({batch_size, npts, 1});
    auto up_i_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_INT64, up_i_tensor_shape);
    // auto up_i_tensor_32 = ::absl::make_unique<tf::Tensor>(tf::DT_INT32, up_i_tensor_shape);

    // std::cout << neigh_idx_tensor_shape << std::endl;
    // std::cout << sub_points_tensor_shape << std::endl;
    // std::cout << pool_i_tensor_shape << std::endl;
    // std::cout << up_i_tensor_shape << std::endl;
    // start to compute
    auto pt_tensor = temp_point_tensor->flat<float>().data();
    auto q_tensor = temp_point_tensor->flat<float>().data();
    auto neigh_idx_flat = neigh_idx_tensor->flat<long long int>().data();
    cpp_knn_batch_omp(pt_tensor, batch_size, npts, dim, q_tensor, nqueries, K_cpp, neigh_idx_flat);
    
    for (int r = 0; r < npts/sub_sampling_ratio[layer] ; ++r) {
      for (int c = 0; c < K_cpp; ++c) {
        pool_i_tensor->tensor<long long int, 3>()(0, r, c) = neigh_idx_tensor->tensor<long long int, 3>()(0, r, c);
        //   if(r ==0 & c == 0){
        //     std::cout << "pool_i_tensor: " <<std::endl;
        //   }
        // // if(r == 0){
        // //   std::cout << "pool_i_tensor: " << std::to_string(pool_i_tensor->tensor<long long int, 3>()(0, r, c)) << std::endl;
        // // }

        //   if (r < 5){
        //     std::cout << std::to_string(pool_i_tensor->tensor<long long int, 3>()(0, c, r)) << "\t";
        //   }
        //   if(c == K_cpp - 1){
        //     std::cout << std::endl;
        // }
      }
    }    

    // std::cout << "subpoint "  << npts/sub_sampling_ratio[layer] << std::endl;
    for (int r = 0; r < npts/sub_sampling_ratio[layer] ; ++r) {
      for (int c = 0; c < dim; ++c) {
        // sub_points_tensor.tensor<float, 3>()(0, r, c) = temp_point_tensor->tensor<float, 3>()(0, r, c);
        sub_points_tensor->tensor<float, 3>()(0, r, c) = temp_point_tensor->tensor<float, 3>()(0, r, c);
        // std::cout << "r " << r << " c " << c << std::endl;
        // std::cout << sub_points_tensor.tensor<float, 3>()(0, r, c) << std::endl;
      }
    }   

    auto sub_points_flat = sub_points_tensor->flat<float>().data();
    auto q_tensor_2 = temp_point_tensor->flat<float>().data();
    auto up_i_flat = up_i_tensor->flat<long long int>().data();
    cpp_knn_batch_omp(sub_points_flat, batch_size, npts/sub_sampling_ratio[layer], dim, q_tensor_2, nqueries, 1, up_i_flat);


    // std::string temp_point_tensor_name = "BATCH_XYZ_" + std::to_string(layer) + "_tensor";

    std::cout << "temp_point_tensor.release()" << OutputTag[15+layer] << std::endl;
    cc->Outputs().Tag(OutputTag[15+layer]).Add(temp_point_tensor.release(), cc->InputTimestamp());

    // std::cout << "delete temp_point_tensor " << std::endl;
    // temp_point_tensor.release();
    // std::cout << "after deleting temp_point_tensor " << std::endl;
    if(layer < 4){
        temp_point_tensor = ::absl::make_unique<tf::Tensor>(tf::DT_FLOAT, sub_points_tensor_shape);
        for (int r = 0; r < npts/sub_sampling_ratio[layer] ; ++r) {
            for (int c = 0; c < dim; ++c) {
                // temp_point_tensor->tensor<float, 3>()(0, r, c) = sub_points_tensor.tensor<float, 3>()(0, r, c);
                temp_point_tensor->tensor<float, 3>()(0, r, c) = sub_points_tensor->tensor<float, 3>()(0, r, c);
                // std::cout << sub_points_tensor.tensor<float, 3>()(0, r, c) << std::endl;
            }
        }
    } 


    // // cast from int63 to int32
    // for (int r = 0; r < npts ; ++r) {
    //     for (int c = 0; c < dim; ++c) {
    //         // temp_point_tensor->tensor<float, 3>()(0, r, c) = sub_points_tensor.tensor<float, 3>()(0, r, c);
    //         up_i_tensor_32->tensor<int, 3>()(0, r, c) = (int) up_i_tensor->tensor<long long int, 3>()(0, r, c);
    //         // std::cout << sub_points_tensor.tensor<float, 3>()(0, r, c) << std::endl;
    //     }
    // }

    // for (int r = 0; r < 50 ; ++r) {
    //   for (int c = 0; c < 1; ++c) {
    //       // temp_point_tensor->tensor<float, 3>()(0, r, c) = sub_points_tensor.tensor<float, 3>()(0, r, c);
    //       // up_i_tensor->tensor<int, 3>()(0, 0, c) = (int) up_i_tensor->tensor<long long int, 3>()(0, 0, c);
    //       // // std::cout << sub_points_tensor.tensor<float, 3>()(0, r, c) << std::endl;
    //       // if(r == 0 & c== 0){
    //       //   std::cout << "up_i_tensor: " ;
    //       // }
          
    //       // std::cout <<  up_i_tensor->tensor<long long int, 3>()(0, r, c) << "\t";
    //       // if(r % 10 == 0){
    //       //   std::cout << std::endl;
    //       // }
    //       // if (r == npts - 1 & c == dim -1){
    //       //   std::cout << std::endl;
    //       // }
    //   }
    // }

    // std::string neigh_idx_tensor_name = "NEIGHBOR_INDEX_" + std::to_string(layer) + "_tensor";

    std::cout << "neigh_idx_tensor.release()" << OutputTag[layer] << std::endl;
    cc->Outputs().Tag(OutputTag[layer]).Add(neigh_idx_tensor.release(), cc->InputTimestamp());

    // std::string sub_points_tensor_name = "SUBPOINTS_" + std::to_string(layer) + "_tensor";

    // std::cout << "sub_points_tensor.release()" << OutputTag[5 +layer]<< std::endl;
    // cc->Outputs().Tag(OutputTag[5 +layer]).Add(sub_points_tensor.release(), cc->InputTimestamp());

    sub_points_tensor.release();

    // std::string pool_i_tensor_name = "POOL_I_" + std::to_string(layer) + "_tensor";

    std::cout << "pool_i_tensor.release()" << OutputTag[5 +layer] << std::endl;
    cc->Outputs().Tag(OutputTag[5 + layer]).Add(pool_i_tensor.release(), cc->InputTimestamp());

    // std::string up_i_tensor_name = "UP_I_" + std::to_string(layer) + "_tensor";

    std::cout << "up_i_tensor.release()" << OutputTag[10 +layer] << std::endl;
    cc->Outputs().Tag(OutputTag[10 +layer]).Add(up_i_tensor.release(), cc->InputTimestamp());
    // neigh_idx_tensor.release();
    // sub_points_tensor.release();
    // pool_i_tensor.release();
    // up_i_tensor.release();
  }
  temp_point_tensor.release();

  std::cout << "DONE POINT CLOUD TO RANDLANET FORMAT CALCULATOR" << std::endl;
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
