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
//

#include "src/core/model_repository_manager.h"

#include <algorithm>
#include <thread>
#include "src/core/backend.h"
#include "src/core/constants.h"
#include "src/core/filesystem.h"
#include "src/core/logging.h"
#include "src/core/model_config_utils.h"
#include "src/core/server_status.h"
#include "src/servables/caffe2/netdef_backend_factory.h"
#include "src/servables/caffe2/netdef_bundle.pb.h"
#include "src/servables/custom/custom_backend_factory.h"
#include "src/servables/custom/custom_bundle.pb.h"
#include "src/servables/ensemble/ensemble_backend_factory.h"
#include "src/servables/ensemble/ensemble_bundle.pb.h"
#include "src/servables/tensorflow/graphdef_backend_factory.h"
#include "src/servables/tensorflow/graphdef_bundle.pb.h"
#include "src/servables/tensorflow/savedmodel_backend_factory.h"
#include "src/servables/tensorflow/savedmodel_bundle.pb.h"
#include "src/servables/tensorrt/plan_backend_factory.h"
#include "src/servables/tensorrt/plan_bundle.pb.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_statistics.h"
#include "tensorflow_serving/config/model_server_config.pb.h"
#include "tensorflow_serving/config/platform_config.pb.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/model_servers/server_core.h"

namespace nvidia { namespace inferenceserver {

struct ModelRepositoryManager::ModelInfo {
  // [TODO] split modification time into versions' and model's
  // so that we have more information on whether the model reload
  // is necessary
  int64_t mtime_nsec_;
  // std::map<int64_t, int64_t> version_mtime_nsec_;
  ModelConfig model_config_;
  tfs::ModelConfig tfs_model_config_;
  Platform platform_;
};

namespace {

void
BuildPlatformConfigMap(
    const std::string& version, const std::string& model_store_path,
    const bool strict_model_config, const float tf_gpu_memory_fraction,
    const bool tf_allow_soft_placement, PlatformConfigMap* platform_configs)
{
  ::google::protobuf::Any graphdef_source_adapter_config;
  ::google::protobuf::Any saved_model_source_adapter_config;
  ::google::protobuf::Any plan_source_adapter_config;
  ::google::protobuf::Any netdef_source_adapter_config;
  ::google::protobuf::Any custom_source_adapter_config;
  ::google::protobuf::Any ensemble_source_adapter_config;

  //// Tensorflow GraphDef
  {
    GraphDefBundleSourceAdapterConfig graphdef_config;

    graphdef_config.set_autofill(!strict_model_config);

    // Tensorflow session config
    if (tf_gpu_memory_fraction == 0.0) {
      graphdef_config.mutable_session_config()
          ->mutable_gpu_options()
          ->set_allow_growth(true);
    } else {
      graphdef_config.mutable_session_config()
          ->mutable_gpu_options()
          ->set_per_process_gpu_memory_fraction(tf_gpu_memory_fraction);
    }

    graphdef_config.mutable_session_config()->set_allow_soft_placement(
        tf_allow_soft_placement);
    graphdef_source_adapter_config.PackFrom(graphdef_config);
  }

  //// Tensorflow SavedModel
  {
    SavedModelBundleSourceAdapterConfig saved_model_config;

    saved_model_config.set_autofill(!strict_model_config);

    if (tf_gpu_memory_fraction == 0.0) {
      saved_model_config.mutable_session_config()
          ->mutable_gpu_options()
          ->set_allow_growth(true);
    } else {
      saved_model_config.mutable_session_config()
          ->mutable_gpu_options()
          ->set_per_process_gpu_memory_fraction(tf_gpu_memory_fraction);
    }

    saved_model_config.mutable_session_config()->set_allow_soft_placement(
        tf_allow_soft_placement);
    saved_model_source_adapter_config.PackFrom(saved_model_config);
  }

  //// Caffe NetDef
  {
    NetDefBundleSourceAdapterConfig netdef_config;
    netdef_config.set_autofill(!strict_model_config);
    netdef_source_adapter_config.PackFrom(netdef_config);
  }

  //// TensorRT
  {
    PlanBundleSourceAdapterConfig plan_config;
    plan_config.set_autofill(!strict_model_config);
    plan_source_adapter_config.PackFrom(plan_config);
  }

  //// Custom
  {
    CustomBundleSourceAdapterConfig custom_config;
    custom_config.set_inference_server_version(version);
    custom_config.set_model_repository_path(model_store_path);
    custom_source_adapter_config.PackFrom(custom_config);
  }

  //// Ensemble
  {
    EnsembleBundleSourceAdapterConfig ensemble_config;
    ensemble_source_adapter_config.PackFrom(ensemble_config);
  }

  (*platform_configs)[kTensorFlowGraphDefPlatform] =
      graphdef_source_adapter_config;
  (*platform_configs)[kTensorFlowSavedModelPlatform] =
      saved_model_source_adapter_config;
  (*platform_configs)[kCaffe2NetDefPlatform] = netdef_source_adapter_config;
  (*platform_configs)[kTensorRTPlanPlatform] = plan_source_adapter_config;
  (*platform_configs)[kCustomPlatform] = custom_source_adapter_config;
  (*platform_configs)[kEnsemblePlatform] = ensemble_source_adapter_config;
}

int64_t
GetModifiedTime(const std::string& path)
{
  // If there is an error in any step the fall-back default
  // modification time is 0. This means that in error cases 'path'
  // will show as not modified. This is the safe fall-back to avoid
  // assuming a model is constantly being modified.

  // If 'path' is a file return its mtime.
  if (!tensorflow::Env::Default()->IsDirectory(path).ok()) {
    tensorflow::FileStatistics stat;
    tensorflow::Status status = tensorflow::Env::Default()->Stat(path, &stat);
    if (!status.ok()) {
      LOG_ERROR << "Failed to determine modification time for '" << path
                << "': " << status;
      return 0;
    }

    return stat.mtime_nsec;
  }

  // 'path' is a directory. Return the most recent mtime of the
  // contents of the directory.
  //
  // GetChildren() returns all descendants instead for cloud storage
  // like GCS.  In such case we should filter out all non-direct
  // descendants.
  std::vector<std::string> children;
  if (!tensorflow::Env::Default()->GetChildren(path, &children).ok()) {
    LOG_ERROR << "Failed to determine modification time for '" << path
              << "', assuming 0";
    return 0;
  }

  std::set<std::string> real_children;
  for (size_t i = 0; i < children.size(); ++i) {
    const std::string& child = children[i];
    real_children.insert(child.substr(0, child.find_first_of('/')));
  }

  // use the modification time of the directory as baseline
  // in case of file deletion
  tensorflow::FileStatistics stat;
  tensorflow::Status status = tensorflow::Env::Default()->Stat(path, &stat);
  if (!status.ok()) {
    LOG_ERROR << "Failed to determine modification time for '" << path
              << "': " << status;
    return 0;
  }
  int64_t mtime = stat.mtime_nsec;

  for (const auto& child : real_children) {
    const auto full_path = tensorflow::io::JoinPath(path, child);
    mtime = std::max(mtime, GetModifiedTime(full_path));
  }

  return mtime;
}

// Return true if any file in the subdirectory root at 'path' has been
// modified more recently than 'last'. Return the most-recent modified
// time in 'last'.
bool
IsModified(const std::string& path, int64_t* last_ns)
{
  const int64_t repo_ns = GetModifiedTime(path);
  bool modified = repo_ns > *last_ns;
  *last_ns = repo_ns;
  return modified;
}

class BackendHandleImpl : public ModelRepositoryManager::BackendHandle {
 public:
  ~BackendHandleImpl() { LOG_INFO << "unload"; OnDestroyBackend_(); LOG_INFO << "unload"; }
  BackendHandleImpl(
      std::unique_ptr<InferenceBackend> is,
      std::function<void()> OnDestroyBackend);
  InferenceBackend* GetInferenceBackend() override { return is_.get(); }

 private:
  std::unique_ptr<InferenceBackend> is_;

  // Use to inform the BackendLifeCycle that the backend handle is destroyed
  std::function<void()> OnDestroyBackend_;
};

BackendHandleImpl::BackendHandleImpl(
    std::unique_ptr<InferenceBackend> is,
    std::function<void()> OnDestroyBackend)
  : is_(std::move(is)), 
    OnDestroyBackend_(std::move(OnDestroyBackend))
{
}

}  // namespace

class ModelRepositoryManager::BackendLifeCycle {
 public:
  static Status Create(
      const PlatformConfigMap& platform_map,
      const std::string& repository_path,
      std::unique_ptr<BackendLifeCycle>* life_cycle);

  // For now, Load() will first unload all versions of the model and then
  // load the requested versions
  Status AsyncLoad(
      const std::string& model_name, const std::vector<int64_t>& versions,
      const ModelConfig& model_config, bool force_unload = true);
  Status Load(
      const std::string& model_name, const int64_t version,
      const ModelConfig& model_config);
  Status Unload(
      const std::string& model_name, const int64_t version);
  Status GetBackendHandle(
      const std::string& model_name, const int64_t version,
      std::shared_ptr<BackendHandle>* handle);
  const ModelMap GetLiveBackendStates();
  const VersionStateMap GetVersionStates(const std::string& model_name);
  
 private:
  struct BackendInfo {
    BackendInfo(
        const ModelReadyState state,
        const ActionType next_action,
        const ModelConfig& model_config)
        : state_(state), next_action_(next_action), model_config_(model_config)
    {
      platform_ = GetPlatform(model_config_.platform());
    }

    Platform platform_;

    std::mutex mtx_;
    ModelReadyState state_;
    ActionType next_action_;
    ModelConfig model_config_;
    std::shared_ptr<BackendHandle> handle_;
  };

  BackendLifeCycle(const std::string& repository_path)
    : repository_path_(repository_path)
  {
  }

  Status CreateBackendHandle(
      const std::string& model_name, const int64_t version,
      BackendInfo* backend_info);
    
  Status TriggerNextAction(
      const std::string& model_name, const int64_t version,
      BackendInfo* backend_info);

  using VersionMap = std::map<int64_t, std::unique_ptr<BackendInfo>>;
  using BackendMap = std::map<std::string, VersionMap>;
  BackendMap map_;
  std::mutex map_mtx_;

  const std::string& repository_path_;
  std::unique_ptr<NetDefBackendFactory> netdef_factory_;
  std::unique_ptr<CustomBackendFactory> custom_factory_;
  std::unique_ptr<EnsembleBackendFactory> ensemble_factory_;
  std::unique_ptr<GraphDefBackendFactory> graphdef_factory_;
  std::unique_ptr<SavedModelBackendFactory> savedmodel_factory_;
  std::unique_ptr<PlanBackendFactory> plan_factory_;
};

Status
ModelRepositoryManager::BackendLifeCycle::Create(
    const PlatformConfigMap& platform_map,
    const std::string& repository_path,
    std::unique_ptr<BackendLifeCycle>* life_cycle)
{
  std::unique_ptr<BackendLifeCycle> local_life_cycle(new BackendLifeCycle(repository_path));

  {
    GraphDefBundleSourceAdapterConfig config;
    platform_map.find(kTensorFlowGraphDefPlatform)->second.UnpackTo(&config);
    RETURN_IF_ERROR(GraphDefBackendFactory::Create(config, &(local_life_cycle->graphdef_factory_)));
  }
  {
    SavedModelBundleSourceAdapterConfig config;
    platform_map.find(kTensorFlowSavedModelPlatform)->second.UnpackTo(&config);
    RETURN_IF_ERROR(SavedModelBackendFactory::Create(config, &(local_life_cycle->savedmodel_factory_)));
  }
  {
    NetDefBundleSourceAdapterConfig config;
    platform_map.find(kCaffe2NetDefPlatform)->second.UnpackTo(&config);
    RETURN_IF_ERROR(NetDefBackendFactory::Create(config, &(local_life_cycle->netdef_factory_)));
  }
  {
    PlanBundleSourceAdapterConfig config;
    platform_map.find(kTensorRTPlanPlatform)->second.UnpackTo(&config);
    RETURN_IF_ERROR(PlanBackendFactory::Create(config, &(local_life_cycle->plan_factory_)));
  }
  {
    CustomBundleSourceAdapterConfig config;
    platform_map.find(kCustomPlatform)->second.UnpackTo(&config);
    RETURN_IF_ERROR(CustomBackendFactory::Create(config, &(local_life_cycle->custom_factory_)));
  }
  {
    EnsembleBundleSourceAdapterConfig config;
    platform_map.find(kEnsemblePlatform)->second.UnpackTo(&config);
    RETURN_IF_ERROR(EnsembleBackendFactory::Create(config, &(local_life_cycle->ensemble_factory_)));
  }

  *life_cycle = std::move(local_life_cycle);
  return Status::Success;
}

const ModelRepositoryManager::ModelMap
ModelRepositoryManager::BackendLifeCycle::GetLiveBackendStates()
{
  std::lock_guard<std::mutex> map_lock(map_mtx_);
  ModelMap live_backend_states;
  for (auto& model_version : map_) {
    bool live = false;
    VersionStateMap version_map;
    for (auto& version_backend : model_version.second) {
      std::lock_guard<std::mutex> lock(version_backend.second->mtx_);
      // At lease one version is live (ready / loading / unloading)
      if ((version_backend.second->state_ != ModelReadyState::MODEL_UNKNOWN)
        && (version_backend.second->state_ != ModelReadyState::MODEL_UNAVAILABLE)) {
        live = true;
        version_map[version_backend.first] = version_backend.second->state_;
      }
    }
    if (live) {
      live_backend_states[model_version.first] = version_map;
    }
  }
  return live_backend_states;
}

const ModelRepositoryManager::VersionStateMap
ModelRepositoryManager::BackendLifeCycle::GetVersionStates(const std::string& model_name)
{
  std::lock_guard<std::mutex> map_lock(map_mtx_);
  VersionStateMap version_map;
  auto mit = map_.find(model_name);
  if (mit != map_.end()) {
    for (auto& version_backend : mit->second) {
      std::lock_guard<std::mutex> lock(version_backend.second->mtx_);
      version_map[version_backend.first] = version_backend.second->state_;
    } 
  }
  
  return version_map;
}

Status
ModelRepositoryManager::BackendLifeCycle::GetBackendHandle(
    const std::string& model_name, const int64_t version,
    std::shared_ptr<BackendHandle>* handle)
{
  std::lock_guard<std::mutex> map_lock(map_mtx_);
  auto mit = map_.find(model_name);
  if (mit == map_.end()) {
    return Status(RequestStatusCode::NOT_FOUND, "model '" + model_name + "' is not found");
  }

  // [TODO] use heap?
  auto vit = mit->second.find(version);
  if (vit == mit->second.end()) {
    // In case the request is asking for latest version
    int64_t latest = -1;
    if (version == -1) {
      for (auto it = mit->second.begin(); it != mit->second.end(); it++) {
        if (it->first > latest) {
          std::lock_guard<std::mutex> lock(it->second->mtx_);
          if (it->second->state_ == ModelReadyState::MODEL_READY) {
            latest = it->first;
            vit = it;
          }
        }
      }
    }
    if (latest == -1) {
      return Status(RequestStatusCode::NOT_FOUND, "model '" + model_name + "' version " + std::to_string(version) + " is not found");
    } else {
      std::lock_guard<std::mutex> lock(vit->second->mtx_);
      *handle = vit->second->handle_;
    }
  } else {
    std::lock_guard<std::mutex> lock(vit->second->mtx_);
    if (vit->second->state_ == ModelReadyState::MODEL_READY) {
      *handle = vit->second->handle_;
    } else {
      return Status(RequestStatusCode::UNAVAILABLE, "model '" + model_name + "' version " + std::to_string(version) + " is not at ready state");
    }
  }
  return Status::Success;
}

Status
ModelRepositoryManager::BackendLifeCycle::Load(
    const std::string& model_name, const int64_t version, const ModelConfig& model_config)
{
  return Status(RequestStatusCode::UNSUPPORTED, "load funtion is not implemented");
  // [TODO] think about how to do it synchronously
  // [TODO] mutex
  // auto it = map_.find(model_name);
  // if (it == map_.end()) {
  //   it = map_.emplace(std::make_pair(model_name, VersionMap())).first;
  // }

  // auto vit = it->second.find(version);
  // if (vit == it->second.end()) {
  //   // [TODO] fix this (see AsyncLoad())
  //   auto backend_state = BackendInfo(ModelReadyState::MODEL_UNKNOWN, ActionType::NO_ACTION, model_config);
  //   vit = it->second.emplace(std::make_pair(version, backend_state)).first;
  // }

  // vit->second->model_config_ = model_config;
  // if (vit->second->state_ == ModelReadyState::READY) {
  //   vit->second->state_ = ModelReadyState::MODEL_UNLOADING;
  //   // [TODO] do something to "sychronize" the unload
  //   // override on destroy callback
  //   vit->second->handle_.reset();
  // }
  // if ((vit->second->state_ == ModelReadyState::MODEL_UNLOADING) || (vit->second->state_ == ModelReadyState::MODEL_LOADING)) {
  //   vit->second->next_action_ = ActionType::LOAD;
  // } else {
  //   // [TODO] detach thread to handle this
  //   CreateBackendHandle(model_name, version, vit->second);
  // }
}

Status
ModelRepositoryManager::BackendLifeCycle::Unload(
    const std::string& model_name, const int64_t version)
{
  return Status(RequestStatusCode::UNSUPPORTED, "unload funtion is not implemented");
  // auto it = map_.find(model_name);
  // if (it == map_.end()) {
  //   return Status(RequestStatusCode::NOT_FOUND, "model '" + model_name + "' is not found");
  // }

  // auto vit = it->second.find(version);
  // if (vit == it->second.end()) {
  //   return Status(RequestStatusCode::NOT_FOUND, "model '" + model_name + "' version " + std::to_string(version) + " is not found");
  // }

  // // ensure the model will always be unloaded regardless of the current state
  // vit->second->next_action_ = ActionType::UNLOAD;

  // if (vit->second->state_ == ModelReadyState::READY) {
  //   vit->second->state_ = ModelReadyState::MODEL_UNLOADING;
  //   vit->second->handle_.reset();
  // } else {
  //   return Status(RequestStatusCode::NOT_FOUND, "tried to unload model '" + model_name + "' version " + std::to_string(version) + " which is at ready state");
  // }
  // return Status::Success;
}

Status
ModelRepositoryManager::BackendLifeCycle::AsyncLoad(
    const std::string& model_name, const std::vector<int64_t>& versions,
    const ModelConfig& model_config, bool force_unload)
{
  std::lock_guard<std::mutex> map_lock(map_mtx_);
  auto it = map_.find(model_name);
  if (it == map_.end()) {
    it = map_.emplace(std::make_pair(model_name, VersionMap())).first;
  }

  if (force_unload) {
    LOG_INFO << "Here";
    for (auto& version_backend : it->second) {
      LOG_INFO << "Unloading: " << model_name << ":" << version_backend.first;
      bool should_unload = false;
      {
        std::lock_guard<std::mutex> lock(version_backend.second->mtx_);
        if (version_backend.second->state_ == ModelReadyState::MODEL_READY) {
          version_backend.second->state_ = ModelReadyState::MODEL_UNLOADING;
          should_unload = true;
        } else {
          version_backend.second->next_action_ = ActionType::UNLOAD;
        }
      }
      if (should_unload) {
        LOG_INFO << "Start unload";
        version_backend.second->handle_.reset();
        LOG_INFO << "unloading...";
      }
    }
  }

  for (const auto& version : versions) {
    auto vit = it->second.find(version);
    if (vit == it->second.end()) {
      vit = it->second.emplace(std::make_pair(version, std::unique_ptr<BackendInfo>())).first;
      vit->second.reset(new BackendInfo(ModelReadyState::MODEL_UNKNOWN, ActionType::NO_ACTION, model_config));
    }

    bool should_unload = false;
    {
      std::lock_guard<std::mutex> lock(vit->second->mtx_);
      vit->second->model_config_ = model_config;
      if (vit->second->state_ == ModelReadyState::MODEL_READY) {
        vit->second->state_ = ModelReadyState::MODEL_UNLOADING;
        should_unload = true;
      }
      if ((vit->second->state_ == ModelReadyState::MODEL_UNLOADING) || (vit->second->state_ == ModelReadyState::MODEL_LOADING)) {
        vit->second->next_action_ = ActionType::LOAD;
      } else {
        // [TODO] clean up the logic here
        vit->second->next_action_ = ActionType::NO_ACTION;
        std::thread worker(&ModelRepositoryManager::BackendLifeCycle::CreateBackendHandle, this, model_name, version, vit->second.get());
        worker.detach();
      }
    }
    if (should_unload) {
      vit->second->handle_.reset();
    }
  }

  LOG_INFO << "End AsyncLoad()";

  return Status::Success;
}

Status
ModelRepositoryManager::BackendLifeCycle::CreateBackendHandle(
    const std::string& model_name, const int64_t version,
    BackendInfo* backend_info)
{
  const auto version_path = tensorflow::io::JoinPath(repository_path_, model_name, std::to_string(version));
  // make copy of the current model config in case model config in backend info
  // is updated (another poll) during the creation of backend handle
  ModelConfig model_config;
  {
    std::lock_guard<std::mutex> lock(backend_info->mtx_);
    model_config = backend_info->model_config_;
  }

  // Create backend
  Status status;
  std::unique_ptr<InferenceBackend> is;
  switch (backend_info->platform_) {
    case Platform::PLATFORM_TENSORFLOW_GRAPHDEF:
      status = graphdef_factory_->CreateBackend(version_path, model_config, &is);
      break;
    case Platform::PLATFORM_TENSORFLOW_SAVEDMODEL:
      status = savedmodel_factory_->CreateBackend(version_path, model_config, &is);
      break;
    case Platform::PLATFORM_TENSORRT_PLAN:
      status = plan_factory_->CreateBackend(version_path, model_config, &is);
      break;
    case Platform::PLATFORM_CAFFE2_NETDEF:
      status = netdef_factory_->CreateBackend(version_path, model_config, &is);
      break;
    case Platform::PLATFORM_CUSTOM:
      status = custom_factory_->CreateBackend(version_path, model_config, &is);
      break;
    case Platform::PLATFORM_ENSEMBLE:
      status = ensemble_factory_->CreateBackend(version_path, model_config, &is);
      break;
    default:
      break;
  }

  // Update backend state
  {
    std::lock_guard<std::mutex> lock(backend_info->mtx_);
    if (status.IsOk()) {
      LOG_INFO << "successfully loaded '" << model_name << "' version " << version;
      backend_info->state_ = ModelReadyState::MODEL_READY;
      // [TODO] verify correctness
      // Load only happened when model unavailble or unknown, handle_ is empty
      LOG_ERROR << "before I don't know what unload";
      backend_info->handle_.reset(new BackendHandleImpl(std::move(is),
          [this, model_name, version, backend_info]() mutable {
            LOG_INFO << "uunloadd";
            {
              std::lock_guard<std::mutex> lock(backend_info->mtx_);
              backend_info->state_ = ModelReadyState::MODEL_UNAVAILABLE;
            }
            // Check if next action is requested
            this->TriggerNextAction(model_name, version, backend_info);
          }));
      LOG_ERROR << "after I don't know what unload";
    } else {
      LOG_ERROR << "failed to load '" << model_name << "' version " << version << ": " << status.AsString();
      backend_info->state_ = ModelReadyState::MODEL_UNAVAILABLE;
      backend_info->handle_.reset();
    }
  }

  // Check if next action is requested
  return TriggerNextAction(model_name, version, backend_info);
}

Status
ModelRepositoryManager::BackendLifeCycle::TriggerNextAction(
      const std::string& model_name, const int64_t version,
      BackendInfo* backend_info)
{
  bool should_unload = false;
  {
    std::lock_guard<std::mutex> lock(backend_info->mtx_);

    switch (backend_info->next_action_) {
      case ActionType::LOAD:
        switch (backend_info->state_) {
          case ModelReadyState::MODEL_READY:
            // unload first, actual loal will be triggered separately after unload
            backend_info->state_ = ModelReadyState::MODEL_UNLOADING;
            should_unload = true;
            break;
          case ModelReadyState::MODEL_UNAVAILABLE:
          // [TODO] should unknown return error?
          case ModelReadyState::MODEL_UNKNOWN:
            backend_info->next_action_ = ActionType::NO_ACTION;
            backend_info->state_ = ModelReadyState::MODEL_LOADING;
            {
              std::thread worker(&ModelRepositoryManager::BackendLifeCycle::CreateBackendHandle, this, model_name, version, backend_info);
              worker.detach();
            }
            break;
          default:
            LOG_ERROR << "unexpecting model state: " << backend_info->state_;
            break;
        }
        break;
      case ActionType::UNLOAD:
        switch (backend_info->state_) {
          case ModelReadyState::MODEL_READY:
            backend_info->next_action_ = ActionType::NO_ACTION;
            backend_info->state_ = ModelReadyState::MODEL_UNLOADING;
            should_unload = true;
            break;
          case ModelReadyState::MODEL_UNAVAILABLE:
          case ModelReadyState::MODEL_UNKNOWN:
            backend_info->next_action_ = ActionType::NO_ACTION;
            break;  
          default:
            LOG_ERROR << "unexpecting model state: " << backend_info->state_;
            break;
        }
        break;
      default:
        break;
    }
  }
  if (should_unload) {
    backend_info->handle_.reset();
  }
  return Status::Success;
}

ModelRepositoryManager* ModelRepositoryManager::singleton = nullptr;

ModelRepositoryManager::ModelRepositoryManager(
    const std::shared_ptr<ServerStatusManager>& status_manager,
    const std::string& repository_path,
    const PlatformConfigMap& platform_config_map, const bool autofill,
    const bool polling_enabled,
    std::unique_ptr<BackendLifeCycle> life_cycle)
    : repository_path_(repository_path),
      platform_config_map_(platform_config_map), autofill_(autofill),
      polling_enabled_(polling_enabled), status_manager_(status_manager),
      backend_life_cycle_(std::move(life_cycle))
{
}

ModelRepositoryManager::~ModelRepositoryManager()
{
  singleton = nullptr;
}

Status
ModelRepositoryManager::Create(
    const std::string& server_version,
    const std::shared_ptr<ServerStatusManager>& status_manager,
    const std::string& repository_path, const bool strict_model_config,
    const float tf_gpu_memory_fraction, const bool tf_allow_soft_placement,
    const uint32_t repository_poll_secs, const bool polling_enabled,
    std::unique_ptr<ModelRepositoryManager>* model_repository_manager)
{
  if (singleton != nullptr) {
    return Status(
        RequestStatusCode::ALREADY_EXISTS,
        "ModelRepositoryManager singleton already created");
  }
  PlatformConfigMap platform_config_map;

  BuildPlatformConfigMap(
      server_version, repository_path, strict_model_config,
      tf_gpu_memory_fraction, tf_allow_soft_placement, &platform_config_map);

  // Not setting the singleton / smart pointer directly because error on TFS
  // core creation may not be considered as initialization failure. So only
  // setting it before core creation to simplify clean up
  std::unique_ptr<BackendLifeCycle> life_cycle;
  RETURN_IF_ERROR(BackendLifeCycle::Create(platform_config_map, repository_path, &life_cycle));
  std::unique_ptr<ModelRepositoryManager> local_manager(
      new ModelRepositoryManager(
          status_manager, repository_path, platform_config_map,
          !strict_model_config, polling_enabled, std::move(life_cycle)));

  // Similar to PollAndUpdate(), but simplier
  std::set<std::string> added, deleted, modified, unmodified;
  if (polling_enabled) {
    RETURN_IF_ERROR(
        local_manager->Poll(&added, &deleted, &modified, &unmodified));
  }
  if (!deleted.empty() || !modified.empty() || !unmodified.empty()) {
    return Status(
        RequestStatusCode::INTERNAL,
        "Unexpected initial state for model repository");
  }

  for (const auto& name : added) {
    ModelConfig model_config;
    RETURN_IF_ERROR(
        local_manager->GetModelConfig(name, &model_config));
    RETURN_IF_ERROR(local_manager->status_manager_->InitForModel(name, model_config));

    std::vector<int64_t> versions;
    RETURN_IF_ERROR(local_manager->VersionsToLoad(name, model_config, versions));

    // We assume that any failure is due to a model not loading correctly
    // so we just continue if not exiting on error.
    local_manager->backend_life_cycle_->AsyncLoad(name, versions, model_config);
  }

  // Create the server core. 
  *model_repository_manager = std::move(local_manager);
  singleton = model_repository_manager->get();

  return Status::Success;
}

Status
ModelRepositoryManager::PollAndUpdate()
{
  if (!polling_enabled_) {
    return Status(RequestStatusCode::INVALID, "polling is disabled");
  }
  std::set<std::string> added, deleted, modified, unmodified;
  RETURN_IF_ERROR(Poll(&added, &deleted, &modified, &unmodified));
  // Nothing to do if no model adds, deletes or modifies.
  if (added.empty() && deleted.empty() && modified.empty()) {
    return Status::Success;
  }

  // Added models should be loaded and be initialized for status
  // reporting.
  for (const auto& name : added) {
    ModelConfig model_config;
    RETURN_IF_ERROR(GetModelConfig(name, &model_config));
    RETURN_IF_ERROR(status_manager_->InitForModel(name, model_config));
    
    std::vector<int64_t> versions;
    RETURN_IF_ERROR(VersionsToLoad(name, model_config, versions));
    backend_life_cycle_->AsyncLoad(name, versions, model_config);
  }

  // If there are any modified model, (re)load them to pick up
  // the changes. We want to keep the current status information
  // so don't re-init it.
  for (const auto& name : modified) {
    ModelConfig model_config;
    RETURN_IF_ERROR(GetModelConfig(name, &model_config));
    RETURN_IF_ERROR(
        status_manager_->UpdateConfigForModel(name, model_config));

    std::vector<int64_t> versions;
    RETURN_IF_ERROR(VersionsToLoad(name, model_config, versions));
    backend_life_cycle_->AsyncLoad(name, versions, model_config);
  }

  for (const auto& name : deleted) {
    ModelConfig model_config;
    std::vector<int64_t> versions;
    // Utilize "force_unload" of AsyncLoad()
    backend_life_cycle_->AsyncLoad(name, versions, model_config);
  }

  return Status::Success;
}

Status
ModelRepositoryManager::LoadUnloadModel(
    const std::string& model_name, ActionType type,
    std::function<void(Status)> OnCompleteUpdate)
{
  if (polling_enabled_) {
    return Status(
        RequestStatusCode::INVALID,
        "explicit model load / unload is not allowed if polling is enabled");
  }
  // [TODO] model load / unload should be done in separate thread
  Status status = Status(RequestStatusCode::UNSUPPORTED, "not implemented");
  OnCompleteUpdate(status);
  return status;
}

Status
ModelRepositoryManager::UnloadAllModels()
{
  Status status;
  // Reload an empty version list to cause the model to unload.
  ModelConfig model_config;
  std::vector<int64_t> versions;
  for (const auto& name_info : infos_) {
    LOG_INFO << "Calling AsyncLoad() for unload";
    Status unload_status = backend_life_cycle_->AsyncLoad(name_info.first, versions, model_config);
    if (!unload_status.IsOk()) {
      status = Status(
          RequestStatusCode::INTERNAL,
          "Failed to gracefully unload models: " + unload_status.Message());
    }
  }
  return Status::Success;
}

const ModelRepositoryManager::ModelMap
ModelRepositoryManager::GetLiveBackendStates()
{
  return backend_life_cycle_->GetLiveBackendStates();
}

const ModelRepositoryManager::VersionStateMap
ModelRepositoryManager::GetVersionStates(const std::string& model_name)
{
  return backend_life_cycle_->GetVersionStates(model_name);
}

Status
ModelRepositoryManager::GetBackendHandle(
    const std::string& model_name, const int64_t model_version,
    std::shared_ptr<BackendHandle>* handle)
{
  Status status = backend_life_cycle_->GetBackendHandle(model_name, model_version, handle);
  if (!status.IsOk()) {
    handle->reset();
    status = Status(
        RequestStatusCode::UNAVAILABLE,
        "Inference request for unknown model '" + model_name + "'");
  }
  return status;
}

Status
ModelRepositoryManager::Poll(
    std::set<std::string>* added, std::set<std::string>* deleted,
    std::set<std::string>* modified, std::set<std::string>* unmodified)
{
  // Serialize all polling operation...
  std::lock_guard<std::mutex> lock(poll_mu_);

  added->clear();
  deleted->clear();
  modified->clear();
  unmodified->clear();

  // We don't modify 'infos_' in place to minimize how long we need to
  // hold the lock and also prevent any partial changes to do an error
  // during processing.
  ModelInfoMap new_infos;

  // Each subdirectory of repository path is a model directory from
  // which we read the model configuration.
  std::vector<std::string> children;
  RETURN_IF_TF_ERROR(
      tensorflow::Env::Default()->GetChildren(repository_path_, &children));

  // GetChildren() returns all descendants instead for cloud storage
  // like GCS.  In such case we should filter out all non-direct
  // descendants.
  std::set<std::string> real_children;
  for (size_t i = 0; i < children.size(); ++i) {
    const std::string& child = children[i];
    real_children.insert(child.substr(0, child.find_first_of('/')));
  }

  for (const auto& child : real_children) {
    const auto full_path = tensorflow::io::JoinPath(repository_path_, child);
    if (!tensorflow::Env::Default()->IsDirectory(full_path).ok()) {
      continue;
    }

    // If 'child' is a new model or an existing model that has been
    // modified since the last time it was polled, then need to
    // (re)load, normalize and validate the configuration.
    bool need_load = false;
    int64_t mtime_ns;
    const auto iitr = infos_.find(child);
    if (iitr == infos_.end()) {
      added->insert(child);
      mtime_ns = GetModifiedTime(std::string(full_path));
      need_load = true;
    } else {
      mtime_ns = iitr->second->mtime_nsec_;
      if (IsModified(std::string(full_path), &mtime_ns)) {
        modified->insert(child);
        need_load = true;
      } else {
        unmodified->insert(child);
        const auto& ret = new_infos.emplace(child, nullptr);
        if (!ret.second) {
          return Status(
              RequestStatusCode::ALREADY_EXISTS,
              "unexpected model info for model '" + child + "'");
        }

        std::unique_ptr<ModelInfo>& model_info = ret.first->second;
        model_info.reset(new ModelInfo(*iitr->second));
      }
    }

    if (need_load) {
      const auto& ret = new_infos.emplace(child, nullptr);
      if (!ret.second) {
        return Status(
            RequestStatusCode::ALREADY_EXISTS,
            "unexpected model info for model '" + child + "'");
      }

      std::unique_ptr<ModelInfo>& model_info = ret.first->second;
      model_info.reset(new ModelInfo());
      ModelConfig& model_config = model_info->model_config_;
      tfs::ModelConfig& tfs_config = model_info->tfs_model_config_;
      model_info->mtime_nsec_ = mtime_ns;

      // If enabled, try to automatically generate missing parts of
      // the model configuration (autofill) from the model
      // definition. In all cases normalize and validate the config.
      RETURN_IF_ERROR(GetNormalizedModelConfig(
          full_path, platform_config_map_, autofill_, &model_config));
      RETURN_IF_ERROR(ValidateModelConfig(model_config, std::string()));

      model_info->platform_ = GetPlatform(model_config.platform());

      // Make sure the name of the model matches the name of the
      // directory. This is a somewhat arbitrary requirement but seems
      // like good practice to require it of the user. It also acts as a
      // check to make sure we don't have two different models with the
      // same name.
      if (model_config.name() != child) {
        return Status(
            RequestStatusCode::INVALID_ARG,
            "unexpected directory name '" + child + "' for model '" +
                model_config.name() +
                "', directory name must equal model name");
      }

      tfs_config.set_name(model_config.name());
      tfs_config.set_base_path(full_path);
      tfs_config.set_model_platform(model_config.platform());

      // Create the appropriate TFS version policy from the model
      // configuration policy.
      if (model_config.version_policy().has_latest()) {
        tfs::FileSystemStoragePathSourceConfig::ServableVersionPolicy::Latest
            latest;
        latest.set_num_versions(
            model_config.version_policy().latest().num_versions());
        tfs_config.mutable_model_version_policy()->mutable_latest()->CopyFrom(
            latest);
      } else if (model_config.version_policy().has_all()) {
        tfs::FileSystemStoragePathSourceConfig::ServableVersionPolicy::All all;
        tfs_config.mutable_model_version_policy()->mutable_all()->CopyFrom(all);
      } else if (model_config.version_policy().has_specific()) {
        tfs::FileSystemStoragePathSourceConfig::ServableVersionPolicy::Specific
            specific;
        specific.mutable_versions()->CopyFrom(
            model_config.version_policy().specific().versions());
        tfs_config.mutable_model_version_policy()->mutable_specific()->CopyFrom(
            specific);
      } else {
        return Status(
            RequestStatusCode::INTERNAL,
            "expected version policy for model '" + model_config.name());
      }
    }
  }

  // Anything in 'infos_' that is not in "added", "modified", or
  // "unmodified" is deleted.
  for (const auto& pr : infos_) {
    if ((added->find(pr.first) == added->end()) &&
        (modified->find(pr.first) == modified->end()) &&
        (unmodified->find(pr.first) == unmodified->end())) {
      deleted->insert(pr.first);
    }
  }

  // Swap the new infos in place under a short-lived lock and only if
  // there were no errors encountered during polling.
  {
    std::lock_guard<std::mutex> lock(infos_mu_);
    infos_.swap(new_infos);
  }

  return Status::Success;
}


Status
ModelRepositoryManager::GetModelConfig(
    const std::string& name, ModelConfig* model_config)
{
  std::lock_guard<std::mutex> lock(infos_mu_);

  const auto itr = infos_.find(name);
  if (itr == infos_.end()) {
    return Status(
        RequestStatusCode::NOT_FOUND,
        "no configuration for model '" + name + "'");
  }

  *model_config = itr->second->model_config_;
  return Status::Success;
}

Status
ModelRepositoryManager::VersionsToLoad(
    const std::string& name, const ModelConfig& model_config,
    std::vector<int64_t>& versions)
{
  versions.clear();

  if (model_config.version_policy().has_specific()) {
    for (const auto& v : model_config.version_policy().specific().versions()) {
      versions.push_back(v);
    }
  } else {
    const auto model_path = tensorflow::io::JoinPath(repository_path_, name);
    std::set<std::string> subdirs;
    RETURN_IF_ERROR(GetSubdirs(model_path, &subdirs));
    
    if (model_config.version_policy().has_latest()) {
      for (const auto& subdir : subdirs) {
        if (versions.size() < model_config.version_policy().latest().num_versions()) {
          versions.push_back(std::stoll(subdir));
        } else {
          auto it = std::min_element(versions.begin(), versions.end());
          int64_t current = std::stoll(subdir);
          if (*it < current) {
            *it = current;
          }
        }
      }
    } else {
      for (const auto& subdir : subdirs) {
        versions.push_back(std::stoll(subdir));
      }
    }
  }

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
