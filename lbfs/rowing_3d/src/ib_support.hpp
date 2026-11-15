#pragma once
#include "numerics.hpp"

// Recomputed once per RK stage; shared by the three IB kernel mappings.
struct PreparedIbMarker { Marker marker; IbStencil stencil; };
