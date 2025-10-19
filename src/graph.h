#pragma once

#include "popnet.h"

#include <vector>

struct popnet_graph
{
  std::vector<popnet_tensor*> nodes;
};
