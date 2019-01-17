/*
 * Getting Started - Following vulkan tutorial to refresh and learn
 * basic triangle application
 *
 * January 2019
 * Wyatt Coppinger
 */


#ifdef NDEBUG
  const bool enable_validation_layers = false;
#else
  const bool enable_validation_layers = true;
#endif


#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <set>
#include <fstream>


const int WIDTH = 800;
const int HEIGHT = 800;
const std::vector<const char*> validation_layers = {
  "VK_LAYER_LUNARG_standard_validation"
};
const std::vector<const char*> device_extensions = {
  VK_KHR_SWAPCHAIN_EXTENSION_NAME
};


//------------------------------------------------------------------------------
// HELPER FUNCTIONS
//------------------------------------------------------------------------------
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GLFW_TRUE);
}


VkResult create_debug_utils_messenger_EXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* p_ci,
    const VkAllocationCallbacks* p_allocator,
    VkDebugUtilsMessengerEXT* p_debug_messenger)
{
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance,
      "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr)
    return func(instance, p_ci, p_allocator, p_debug_messenger);
  else
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}


void destroy_debug_utils_messenger_EXT(VkInstance instance,
    VkDebugUtilsMessengerEXT debug_messenger,
    const VkAllocationCallbacks* p_allocator)
{
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance,
      "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr)
    func(instance, debug_messenger, p_allocator);
}


static std::vector<char> read_file(const std::string& filename)
{
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open())
    throw std::runtime_error("Failed to open file!");

  size_t file_size = (size_t) file.tellg();
  std::vector<char> buffer(file_size);
  file.seekg(0);
  file.read(buffer.data(), file_size);
  file.close();

  return buffer;
}


//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
// STRUCTS
//------------------------------------------------------------------------------
struct QueueFamilyIndices
{
  uint32_t graphics_family = -1;
  uint32_t present_family  = -1;

  bool is_complete() {
    return graphics_family >= 0 && present_family >= 0;
  }
};

struct SwapchainSupportDetails
{
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
// APPLICATION CLASS
//------------------------------------------------------------------------------
class TriangleApp
{
public:
  void run() {
    init_window();
    init_vulkan();
    main_loop();
    cleanup();
  }


private:
  GLFWwindow* window;
  VkInstance instance;
  VkDebugUtilsMessengerEXT debug_messenger;
  VkSurfaceKHR surface;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device;
  VkQueue graphics_queue;
  VkQueue present_queue;
  VkSwapchainKHR swapchain;
  std::vector<VkImage> swapchain_images;
  VkFormat swapchain_image_format;
  VkExtent2D swapchain_extent;
  std::vector<VkImageView> swapchain_image_views;
  VkRenderPass render_pass;
  VkPipelineLayout pipeline_layout;


  std::vector<const char*> get_required_extensions() {
    uint32_t glfw_extension_count = 0;
    const char** glfw_extensions;
    glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

    std::vector<const char*> extensions(glfw_extensions, glfw_extensions + glfw_extension_count);

    if (enable_validation_layers)
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    return extensions;
  }


  /* this is only one of many different ways to setup your validation layers
   * and debug callbacks, have fun figuring out what is best for your
   * specific scenario!
   */
  void setup_debug_messenger()
  {
    if (!enable_validation_layers)
      return;

    VkDebugUtilsMessengerCreateInfoEXT ci = {};
    ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    /* allows you to specify all types of severities you would like callback
     * to be called for
     */
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    /* allows you to filter which types of messages the callback is notified
     * for
     * if the messages aren't useful you can just remove them from the list
     */
    ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    /* specifies pointer to callback function */
    ci.pfnUserCallback = debug_callback;
    ci.pUserData = nullptr; // Optional

    if (create_debug_utils_messenger_EXT(instance, &ci, nullptr, &debug_messenger) != VK_SUCCESS)
      throw std::runtime_error("failed to set up debug messenger!");
  }


  QueueFamilyIndices find_queue_families(VkPhysicalDevice device)
  {
    QueueFamilyIndices indices;
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

    int i = 0;
    for (const auto& queue_family : queue_families) {
      if (queue_family.queueCount > 0 && queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT)
        indices.graphics_family = i;

      VkBool32 present_support = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_support);

      if (queue_family.queueCount > 0 && present_support)
        indices.present_family = i;

      if (indices.is_complete())
        break;

      i++;
    }

    return indices;
  }


  SwapchainSupportDetails query_swapchain_support(VkPhysicalDevice device) {
    SwapchainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t format_count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nullptr);

    if (format_count != 0) {
      details.formats.resize(format_count);
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, details.formats.data());
    }

    uint32_t present_mode_count;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, nullptr);

    if (present_mode_count != 0) {
      details.presentModes.resize(present_mode_count);
      vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, details.presentModes.data());
    }

    return details;
  }


  void create_instance()
  {
    if (enable_validation_layers && !check_validation_layer_support())
      throw std::runtime_error("validation layers requested, but not available!");

    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Getting Started";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "No Engine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app_info;

    auto extensions = get_required_extensions();
    ci.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();

    if (enable_validation_layers) {
      ci.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
      ci.ppEnabledLayerNames = validation_layers.data();
    } else {
      ci.enabledLayerCount = 0;
    }

    if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS)
      throw std::runtime_error("failed to create instance!");

    /* print supported extension names */
    uint32_t supp_extension_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &supp_extension_count, nullptr);

    std::vector<VkExtensionProperties> supp_extensions(supp_extension_count);

    vkEnumerateInstanceExtensionProperties(nullptr, &supp_extension_count, supp_extensions.data());

    std::cout << "available extensions:" << std::endl;

    for (const auto& extension : supp_extensions) {
      std::cout << "\t" << extension.extensionName << std::endl;
    }
  }


  void init_window()
  {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

    glfwSetKeyCallback(window, key_callback);
  }


  void init_vulkan()
  {
    create_instance();
    setup_debug_messenger();
    create_surface();
    pick_physical_device();
    create_logical_device();
    create_swapchain();
    create_image_views();
    create_render_pass();
    create_graphics_pipeline();
  }


  void create_render_pass()
  {
    VkAttachmentDescription color_attachment = {};
    color_attachment.format  = swapchain_image_format;
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    /* applies to color and depth data */
    color_attachment.loadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    /* applies to stencil data */
    color_attachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    /* pixel format of textures and framebuffers can change based on what we're
       doing */
    color_attachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_attachment_ref = {};
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments    = &color_attachment_ref;

    VkRenderPassCreateInfo render_pass_ci = {};
    render_pass_ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_ci.attachmentCount = 1;
    render_pass_ci.pAttachments = &color_attachment;
    render_pass_ci.subpassCount = 1;
    render_pass_ci.pSubpasses   = &subpass;

    if (vkCreateRenderPass(device, &render_pass_ci, nullptr, &render_pass) != VK_SUCCESS)
      throw std::runtime_error("Failed to create render pass!");
  }


  void create_graphics_pipeline()
  {
    auto vert_shader_code = read_file("shaders/vert.spv");
    auto frag_shader_code = read_file("shaders/frag.spv");

    VkShaderModule vert_shader_module = create_shader_module(vert_shader_code);
    VkShaderModule frag_shader_module = create_shader_module(frag_shader_code);

    VkPipelineShaderStageCreateInfo vert_shader_stage_ci = {};
    vert_shader_stage_ci.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vert_shader_stage_ci.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    vert_shader_stage_ci.module = vert_shader_module;
    vert_shader_stage_ci.pName  = "main";

    VkPipelineShaderStageCreateInfo frag_shader_stage_ci = {};
    frag_shader_stage_ci.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    frag_shader_stage_ci.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_shader_stage_ci.module = frag_shader_module;
    frag_shader_stage_ci.pName  = "main";

    VkPipelineShaderStageCreateInfo shader_stages[] = {vert_shader_stage_ci, frag_shader_stage_ci};

    VkPipelineVertexInputStateCreateInfo vert_input_ci = {};
    vert_input_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vert_input_ci.vertexBindingDescriptionCount   = 0;
    vert_input_ci.pVertexBindingDescriptions      = nullptr;
    vert_input_ci.vertexAttributeDescriptionCount = 0;
    vert_input_ci.pVertexAttributeDescriptions    = nullptr;

    VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width  = (float) swapchain_extent.width;
    viewport.height = (float) swapchain_extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = swapchain_extent;

    VkPipelineViewportStateCreateInfo viewport_state = {};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.pViewports    = &viewport;
    viewport_state.scissorCount  = 1;
    viewport_state.pScissors     = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth   = 1.0f;
    rasterizer.cullMode    = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace   = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f;
    rasterizer.depthBiasClamp          = 0.0f;
    rasterizer.depthBiasSlopeFactor    = 0.0f;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable  = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading     = 1.0f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable      = VK_FALSE;

    VkPipelineColorBlendAttachmentState color_blend_attachment = {};
    color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
    VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    color_blend_attachment.blendEnable = VK_FALSE;
    color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo color_blending = {};
    color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.logicOpEnable = VK_FALSE;
    color_blending.logicOp = VK_LOGIC_OP_COPY;
    color_blending.attachmentCount = 1;
    color_blending.pAttachments = &color_blend_attachment;
    color_blending.blendConstants[0] = 0.0f;
    color_blending.blendConstants[1] = 0.0f;
    color_blending.blendConstants[2] = 0.0f;
    color_blending.blendConstants[3] = 0.0f;

    VkDynamicState dynamic_states[] = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_LINE_WIDTH
    };

    VkPipelineDynamicStateCreateInfo dynamic_state = {};
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount = 2;
    dynamic_state.pDynamicStates = dynamic_states;

    VkPipelineLayoutCreateInfo pipeline_layout_ci = {};
    pipeline_layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_ci.setLayoutCount = 0;
    pipeline_layout_ci.pSetLayouts = nullptr;
    pipeline_layout_ci.pushConstantRangeCount = 0;
    pipeline_layout_ci.pPushConstantRanges = nullptr;

    if (vkCreatePipelineLayout(device, &pipeline_layout_ci, nullptr, &pipeline_layout) != VK_SUCCESS)
      throw std::runtime_error("Failed to create pipeline layout!");

    vkDestroyShaderModule(device, frag_shader_module, nullptr);
    vkDestroyShaderModule(device, vert_shader_module, nullptr);
  }


  VkShaderModule create_shader_module(const std::vector<char>& code)
  {
    VkShaderModuleCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule shader_module;
    if (vkCreateShaderModule(device, &ci, nullptr, &shader_module) != VK_SUCCESS)
      throw std::runtime_error("Failed to create a shader module!");

    return shader_module;
  }


  void create_image_views()
  {
    swapchain_image_views.resize(swapchain_images.size());
    for (size_t i = 0; i < swapchain_images.size(); i++) {
      VkImageViewCreateInfo ci = {};
      ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      ci.image = swapchain_images[i];
      ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
      ci.format = swapchain_image_format;
      ci.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
      ci.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
      ci.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
      ci.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
      ci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      ci.subresourceRange.baseMipLevel   = 0;
      ci.subresourceRange.levelCount     = 1;
      ci.subresourceRange.baseArrayLayer = 0;
      ci.subresourceRange.layerCount     = 1;
      if (vkCreateImageView(device, &ci, nullptr, &swapchain_image_views[i]) != VK_SUCCESS)
        throw std::runtime_error("Failed to create image views!");
    }

  }


  void create_swapchain()
  {
    SwapchainSupportDetails swapchain_support = query_swapchain_support(physical_device);

    VkSurfaceFormatKHR surface_format = choose_swap_surface_format(swapchain_support.formats);
    VkPresentModeKHR present_mode = choose_swap_present_mode(swapchain_support.presentModes);
    VkExtent2D extent = choose_swap_extent(swapchain_support.capabilities);

    uint32_t image_count = swapchain_support.capabilities.minImageCount + 1;
    if (swapchain_support.capabilities.maxImageCount > 0 && image_count > swapchain_support.capabilities.maxImageCount)
      image_count = swapchain_support.capabilities.maxImageCount;

    VkSwapchainCreateInfoKHR ci = {};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = surface;
    ci.minImageCount = image_count;
    ci.imageFormat = surface_format.format;
    ci.imageColorSpace = surface_format.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices indices = find_queue_families(physical_device);
    uint32_t queue_family_indices[] = {indices.graphics_family, indices.present_family};

    if (indices.graphics_family != indices.present_family) {
      ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      ci.queueFamilyIndexCount = 2;
      ci.pQueueFamilyIndices = queue_family_indices;
    } else {
      ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
      ci.queueFamilyIndexCount = 0;
      ci.pQueueFamilyIndices = nullptr;
    }

    ci.preTransform = swapchain_support.capabilities.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = present_mode;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(device, &ci, nullptr, &swapchain) != VK_SUCCESS)
      throw std::runtime_error("Failed to create swapchain!");

    vkGetSwapchainImagesKHR(device, swapchain, &image_count, nullptr);
    swapchain_images.resize(image_count);
    vkGetSwapchainImagesKHR(device, swapchain, &image_count, swapchain_images.data());

    swapchain_image_format = surface_format.format;
    swapchain_extent = extent;
  }


  VkPresentModeKHR choose_swap_present_mode(const std::vector<VkPresentModeKHR> available_present_modes)
  {
    VkPresentModeKHR best_mode = VK_PRESENT_MODE_FIFO_KHR;

    for (const auto& available_present_mode : available_present_modes) {
      if (available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR)
        return available_present_mode;
      else if (available_present_mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
        best_mode = available_present_mode;
    }

    return best_mode;
  }


  VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities)
  {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
      return capabilities.currentExtent;
    } else {
      VkExtent2D actual_extent = {WIDTH, HEIGHT};

      actual_extent.width  = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actual_extent.width));
      actual_extent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actual_extent.height));

      return actual_extent;
    }
  }


  VkSurfaceFormatKHR choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& available_formats)
  {
    if (available_formats.size() == 1 && available_formats[0].format == VK_FORMAT_UNDEFINED)
      return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};

    for (const auto& available_format : available_formats) {
      if (available_format.format == VK_FORMAT_B8G8R8A8_UNORM && available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        return available_format;
    }

    return available_formats[0];
  }


  void create_surface()
  {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
      throw std::runtime_error("Failed to create a window surface!");
  }


  void create_logical_device()
  {
    QueueFamilyIndices indices = find_queue_families(physical_device);

    std::vector<VkDeviceQueueCreateInfo> q_cis;
    std::set<uint32_t> unique_queue_families = {indices.graphics_family, indices.present_family};

    float queue_priority = 1.0f;
    for (uint32_t queue_family : unique_queue_families) {
      VkDeviceQueueCreateInfo q_ci = {};
      q_ci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      q_ci.queueFamilyIndex = queue_family;
      q_ci.queueCount = 1;
      q_ci.pQueuePriorities = &queue_priority;
      q_cis.push_back(q_ci);
    }

    VkPhysicalDeviceFeatures device_features = {};

    VkDeviceCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    ci.queueCreateInfoCount = static_cast<uint32_t>(q_cis.size());
    ci.pQueueCreateInfos = q_cis.data();
    ci.pEnabledFeatures = &device_features;

    ci.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
    ci.ppEnabledExtensionNames = device_extensions.data();

    if (enable_validation_layers) {
      ci.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
      ci.ppEnabledLayerNames = validation_layers.data();
    } else {
      ci.enabledLayerCount = 0;
    }

    if (vkCreateDevice(physical_device, &ci, nullptr, &device) != VK_SUCCESS)
      throw std::runtime_error("Failed to create logical device!");

    vkGetDeviceQueue(device, indices.graphics_family, 0, &graphics_queue);
    vkGetDeviceQueue(device, indices.present_family, 0, &present_queue);
  }


  void pick_physical_device()
  {
    uint32_t device_count = 0;

    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);

    if (device_count == 0)
      throw std::runtime_error("Failed to find GPU's with Vulkan support!");

    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

    for (const auto& device : devices) {
      if (is_device_suitable(device)) {
        physical_device = device;
        break;
      }
    }

    if (physical_device == VK_NULL_HANDLE)
      throw std::runtime_error("Failed to find a suitable GPU!");
  }


  bool is_device_suitable(VkPhysicalDevice device)
  {
    QueueFamilyIndices indices = find_queue_families(device);

    bool extensions_supported = check_device_extension_support(device);

    bool swapchain_adequate = false;
    if (extensions_supported) {
      SwapchainSupportDetails swapchain_support = query_swapchain_support(device);
      swapchain_adequate = !swapchain_support.formats.empty() && !swapchain_support.presentModes.empty();
    }

    return indices.is_complete() && extensions_supported && swapchain_adequate;
  }


  /* wow, BEAUTIFUL code! */
  bool check_device_extension_support(VkPhysicalDevice device)
  {
    uint32_t extension_count;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);

    std::vector<VkExtensionProperties> available_extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, available_extensions.data());

    std::set<std::string> required_extensions(device_extensions.begin(), device_extensions.end());

    for (const auto& extension : available_extensions) {
      required_extensions.erase(extension.extensionName);
    }

    return required_extensions.empty();
  }


  void main_loop()
  {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
    }
  }


  void cleanup()
  {
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyRenderPass(device, render_pass, nullptr);
    for (auto image_view : swapchain_image_views) {
      vkDestroyImageView(device, image_view, nullptr);
    }
    vkDestroySwapchainKHR(device, swapchain, nullptr);
    vkDestroyDevice(device, nullptr);
    if (enable_validation_layers)
      destroy_debug_utils_messenger_EXT(instance, debug_messenger, nullptr);

    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();
  }


  bool check_validation_layer_support()
  {
    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

    std::vector<VkLayerProperties> available_layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

    for (const char* layer_name : validation_layers) {
      bool layer_found = false;

      for (const auto& layer_properties : available_layers) {
        if (strcmp(layer_name, layer_properties.layerName) == 0) {
          layer_found = true;
          break;
        }
      }

      if (!layer_found)
        return false;
    }

    return true;
  }

  static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_type,
    const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data,
    void* p_user_data)
  {
    std::cerr << "validation layer: " << p_callback_data->pMessage << std::endl;

    return VK_FALSE;
  }
};
//------------------------------------------------------------------------------




//------------------------------------------------------------------------------
// MAIN FUNCTION
//------------------------------------------------------------------------------
int main()
{
  TriangleApp app;

  try {
    app.run();
  } catch (const std::exception& e) {
      std::cerr << e.what() << std::endl;
      return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
//------------------------------------------------------------------------------
