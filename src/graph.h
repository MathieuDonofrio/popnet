#pragma once

#include <vector>

#include "popnet.h"

struct popnet_graph
{
  std::vector<popnet_tensor*> nodes;
};
