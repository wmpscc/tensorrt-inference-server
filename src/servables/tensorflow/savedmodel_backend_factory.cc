// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "src/servables/tensorflow/savedmodel_backend_factory.h"

#include <memory>
#include <string>
#include <vector>

#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.pb.h"
#include "src/core/model_config_utils.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"

namespace nvidia { namespace inferenceserver {

Status
SavedModelBackendFactory::Create(
    const SavedModelBundleSourceAdapterConfig& platform_config,
    std::unique_ptr<SavedModelBackendFactory>* factory)
{
  LOG_VERBOSE(1) << "Create SavedModelBackendFactory for platform config \""
                 << platform_config.DebugString() << "\"";

  factory->reset(new SavedModelBackendFactory(platform_config));
  return Status::Success;
}

Status
SavedModelBackendFactory::CreateBackend(
      const std::string& path, const ModelConfig& model_config,
      std::unique_ptr<InferenceBackend>* backend)
{
  // Read all the savedmodel directories in 'path'. GetChildren()
  // returns all descendants instead for cloud storage like GCS, so
  // filter out all non-direct descendants.
  std::vector<std::string> possible_children;
  RETURN_IF_TF_ERROR(
      tensorflow::Env::Default()->GetChildren(path, &possible_children));
  std::set<std::string> children;
  for (const auto& child : possible_children) {
    children.insert(child.substr(0, child.find_first_of('/')));
  }

  std::unordered_map<std::string, std::string> savedmodel_paths;
  for (const auto& filename : children) {
    const auto savedmodel_path = tensorflow::io::JoinPath(path, filename);
    if (tensorflow::Env::Default()->IsDirectory(savedmodel_path).ok()) {
      savedmodel_paths.emplace(
          std::piecewise_construct, std::make_tuple(filename),
          std::make_tuple(savedmodel_path));
    }
  }

  std::unique_ptr<SavedModelBundle> local_bundle(new SavedModelBundle);
  RETURN_IF_ERROR(local_bundle->Init(path, model_config));
  RETURN_IF_ERROR(local_bundle->CreateExecutionContexts(
      platform_config_.session_config(), savedmodel_paths));
  
  *backend = std::move(local_bundle);
  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
