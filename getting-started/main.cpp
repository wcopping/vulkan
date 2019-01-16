/*
 * Getting Started - Following vulkan tutorial to refresh and learn
 * basic triangle application
 *
 * January 2019
 * Wyatt Coppinger
 */

/*
 * window surface
 * --------------
 * Vulkan is platform agnostic and therefore does not know what window it will
 * be interfacing with.
 *   We need to use the WSI (window system integration) extensions
 *   We need to use VK_KHR_SURFACE which exposes a VkSurfaceKHR object
 *     represents abstract type of surface to present rendered images to
 *   It is automatically got by glfwGetRequiredInstanceExtensions along with a
 *   few other necessary extensions in the list provided by that function
 *
 * Vulkan does not REQUIRE a window surface (unlike OpenGL which requires at
 * least an invisible one), meaning you can run a graphics program just for the
 * compute ability or off screen rendering from the graphics API .
 *
 * You must ensure that both your Vulkan implementation and your DEVICE support
 * graphics presentation
 *   Therefore you must check that graphics presentation is supported by some
 *   queue family you are using
 *   Drawing and presenting are not bound to same queue families and you must
 *   ensure that both are supported to move forward
 *   Or you could explicitly select a device that has both in the same family
 *
 * presentation queue
 * ------------------
 * We must modify logical device creation to create presentation queue
 *
 * swapchain
 * ---------
 * Vulkan has no default framebuffer
 * We instead have a swap chain which owns the buffers we render to before we
 * visualize them on the screen
 *   This must be explicitly created
 *   Essentially is a queue of images waiting to be presented to the screen
 *   General purpose is synchronize image presentation with screen refresh rate
 *   We can control how the queue works and conditions through set up of the
 *   swapchain
 * We must check for swapchain support
 *
 * Setting up the swapchain is more involved than the physical or logical device
 *
 * core components of swapchain are the PHYSICAL DEVICE and the WINDOW SURFACE
 *
 * must determine three settings:
 *   surface format (color depth)
 *   presentation mode (conditions for "swapping" images to the screen)
 *   swap extent (resolution of images in swapchain)
 *     almost always equal to resolution of windowe we're drawing to
 *     range is defined in VkSurfaceCapabilitiesKHR structure
 *
 * each VkSurfaceFormatKHR contains a format and a colorSpace member
 *   format indicates color channels and types
 *   colorSpace indicates if SRGB color space is supported
 *
 * Arguably most important setting for swapchain is "PRESENTATION MODE"
 *   It represents actual conditions for showing images to the screen
 *
 * Now that we have made all the necessary helper functions:
 *   choose_swap_surface_format
 *   choose_swap_present_mode
 *   choose_swap_extent
 * We can put them together in create_swapchain
 *
 * NOTE ON POST PROCESSING!!
 * in create_swapchain function
 *   VkSwapChainCreateInfo data type has bit field "imageUsage"
 *   This field specifies what kind of operations we'll use the images in the
 *   swap chain for. E.g. POST PROCESSING! we would use
 *   VK_IMAGE_USAGE_TRANSFER_DST_BIT to transfer the image to another image
 *   then use a memory operation to transfer the rendered image to the swapchain
 *   image
 *
 * NOTE:
 * function check_device_extension_support is a beautiful piece of code in my
 * opinion
 *   It uses vector member functions I don't usually see (erase and empty)
 */


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


const int WIDTH = 800;
const int HEIGHT = 800;

const std::vector<const char*> validation_layers = {
  "VK_LAYER_LUNARG_standard_validation"
};

const std::vector<const char*> device_extensions = {
  VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
  const bool enable_validation_layers = false;
#else
  const bool enable_validation_layers = true;
#endif


static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
VkResult create_debug_utils_messenger_EXT(
    VkInstance,
    const VkDebugUtilsMessengerCreateInfoEXT*,
    const VkAllocationCallbacks*,
    VkDebugUtilsMessengerEXT*);


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
    /* GLFW originally inteded for OpenGL
     * we must hint to it that we are not using OpenGL
     */
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
