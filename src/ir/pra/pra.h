#pragma once

// Convenience umbrella for consumers that intentionally opt into the concrete
// PRA IR contract. Generic orchestration should continue to use ir::handle.
#include "ir/pra/builder.h"
#include "ir/pra/descriptor.h"
#include "ir/pra/event.h"
#include "ir/pra/expression.h"
#include "ir/pra/fault_tree.h"
#include "ir/pra/gate.h"
#include "ir/pra/id.h"
#include "ir/pra/metadata.h"
#include "ir/pra/model.h"
#include "ir/pra/target.h"
