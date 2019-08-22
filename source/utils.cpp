#include "utils.hpp"

double stime(){
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);
  double mlsec = 1000.0 * ((double)tv.tv_sec + (double)tv.tv_usec/1000000.0);
  return mlsec/1000.0;
}

void show_params( uint n_runs, uint NP, uint n_evals, uint D, uint PL, std::string FuncObj ){
  int NBA = (NP%32)? (NP/32)+1 : NP/32;
  int NTB = 32 * ceil((double) D / 32.0);

  printf(" | Number of Executions:                    %d\n", n_runs);
  printf(" | Population Size:                         %d\n", NP);
  printf(" | Protein Length:                          %d\n", PL);
  printf(" | Number of Dimensions:                    %d\n", D);
  printf(" | Number of Function Evaluations:          %d\n", n_evals);
  printf(" | Optimization Function:                   %s\n", FuncObj.c_str());
  printf(" +==============================================================+ \n");
  printf(" | Structure (A)\n");
  printf(" | \t Number of Threads                        %d\n", 32);
  printf(" | \t Number of Blocks                         %d\n", NBA);
  printf(" | Structure (B)\n");
  printf(" | \t Number of Threads                        %d\n", NTB);
  printf(" | \t Number of Blocks                         %d\n", NP);
}

std::string toString(uint id){
  switch( id ){
    case 1001:
      return "2D-AB";
    case 1002:
      return "3D-AB";
    default:
      return "Unknown";
  }
}

void save_json(std::vector< std::tuple<uint, float, float, float, float > >& data, std::ofstream& ofs  ){

  Json::Value g1(Json::arrayValue);
  Json::Value y1(Json::arrayValue);

  for( auto it = data.begin(); it != data.end(); it++ ){
    g1.append( std::get<0>(*it) );
    y1.append( std::get<1>(*it) );
  }

  Json::Value event;
  Json::Value conjunto(Json::arrayValue);
  Json::Value go;

  //first
  go["type"] = "scatter";
  go["line"]["color"] = "rgb(24, 0, 243)";
  go["mode"] = "lines";
  go["name"] = "best";
  go["x"] = g1; //<valores de x do best>; // geração
  go["y"] = y1; //<valores de y do best>; // valores best de cada geração

  conjunto.append(go);
  go.clear();

  event["data"] = conjunto;

  // layout config
  event["layout"]["font"]["size"] = 16;
  event["layout"]["font"]["color"] = "rgb(0, 0, 0)";
  event["layout"]["title"]["text"] = "";

  event["layout"]["xaxis"]["type"] = "linear";
  event["layout"]["xaxis"]["title"]["text"] = "Iteration";
  event["layout"]["xaxis"]["mirror"] = "ticks";
  event["layout"]["xaxis"]["showline"] = true;
  event["layout"]["xaxis"]["autorange"] = true;
  event["layout"]["xaxis"]["linecolor"] = "rgb(0, 0, 0)";

  event["layout"]["yaxis"]["type"] = "linear";
  event["layout"]["yaxis"]["title"]["text"] = "Energy Value";
  event["layout"]["yaxis"]["mirror"] = "ticks";
  event["layout"]["yaxis"]["showline"] = true;
  event["layout"]["yaxis"]["autorange"] = true;
  event["layout"]["yaxis"]["linecolor"] = "rgb(0, 0, 0)";

  event["layout"]["legend"]["x"] = 1;
  event["layout"]["legend"]["y"] = 1;
  event["layout"]["legend"]["xanchor"] = "auto";
  event["layout"]["legend"]["borderwidth"] = 0;

  event["layout"]["autosize"] = true;
  event["layout"]["dragmode"] = "pan";

  // Json::FastWriter fast;
  //
  // Json::StyledWriter styled;
  //
  // std::string sFast = fast.write(event);
  // std::string sStyled = styled.write(event);
  //
  // std::cout << "Normal\n" << event << "Fast\n" << sFast << "Styled:\n" << sStyled << std::endl;

  Json::StyledStreamWriter styledStream;
  styledStream.write(ofs, event);
}

void save_diversity(
  std::vector< std::tuple<uint, float, float, float, float > >& data,
  std::ofstream& ofs
){
  Json::Value g1(Json::arrayValue);
  Json::Value y1(Json::arrayValue);

  for( auto it = data.begin(); it != data.end(); it++ ){
    g1.append( std::get<0>(*it) );
    y1.append( std::get<4>(*it) );
  }

  Json::Value event;
  Json::Value conjunto(Json::arrayValue);
  Json::Value go;

  //first
  go["type"] = "scatter";
  go["line"]["color"] = "rgb(5, 122, 12)";
  go["line"]["dash"] = "solid";
  go["line"]["width"] = 3;
  go["mode"] = "lines";
  go["x"] = g1;
  go["y"] = y1;

  conjunto.append(go);
  go.clear();

  event["data"] = conjunto;

  // layout config
  event["layout"]["font"]["size"] = 16;
  event["layout"]["font"]["color"] = "rgb(0, 0, 0)";
  event["layout"]["title"]["text"] = "";

  event["layout"]["xaxis"]["type"] = "linear";
  event["layout"]["xaxis"]["title"]["text"] = "Iteration";
  event["layout"]["xaxis"]["mirror"] = "ticks";
  event["layout"]["xaxis"]["showline"] = true;
  event["layout"]["xaxis"]["autorange"] = true;
  event["layout"]["xaxis"]["gridwidth"] = 2;
  event["layout"]["xaxis"]["linecolor"] = "rgb(0, 0, 0)";

  event["layout"]["yaxis"]["type"] = "linear";
  event["layout"]["yaxis"]["title"]["text"] = "Diversity";
  event["layout"]["yaxis"]["mirror"] = "ticks";
  event["layout"]["yaxis"]["showline"] = true;
  event["layout"]["yaxis"]["autorange"] = true;
  event["layout"]["yaxis"]["linecolor"] = "rgb(0, 0, 0)";

  event["layout"]["autosize"] = true;
  event["layout"]["showlegend"] = false;

  Json::StyledStreamWriter styledStream;
  styledStream.write(ofs, event);
}
