/**
 * @file descriptor.h
 * @brief What a layer manager reports about how it runs: descriptions of the device, the execution plan and its
 *        partitions, the PDAG's nodes, and the kernel DAG
 * @author Arjun Earthperson
 * @date 2026
 *
 * @details Descriptions are plain data for logs and bindings, made after the kernels are built and never read by them.
 * A description holds what is known and computes what follows from it (partition::work_groups and the like), so no
 * two fields can disagree. Nothing here reads the PDAG: the layer manager describes the PDAG's nodes, and the builder
 * (builder/) the partitions it makes.
 *
 *     range.h           range3: a three-dimensional range
 *     device.h          device_profile: a device's identity, limits and memory, and the names of SYCL's enumerations
 *     partition.h       partition: one kernel, its nodes, the partitions it reads, its launch and its gates' fan-in
 *     plan.h            plan: the partitions of a pass in submission order, and the edges they imply
 *     node.h            node: a PDAG node, what it is and which partition computes it
 *     kernel_dag.h      kernel_dag: the nodes grouped per layer by the kind of kernel that computes them
 *     execution_plan.h  execution_plan: all of the above for one pass, as its schema reports it
 */

#pragma once

#include "mc/executor/descriptor/device.h"
#include "mc/executor/descriptor/execution_plan.h"
#include "mc/executor/descriptor/kernel_dag.h"
#include "mc/executor/descriptor/node.h"
#include "mc/executor/descriptor/partition.h"
#include "mc/executor/descriptor/plan.h"
#include "mc/executor/descriptor/range.h"
