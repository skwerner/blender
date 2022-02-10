/*
 * Copyright 2011-2020 Blender Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __IMAGE_OIIO__
#define __IMAGE_OIIO__

#include "scene/image.h"

CCL_NAMESPACE_BEGIN

class OIIOImageLoader : public ImageLoader {
 public:
  OIIOImageLoader(const string &filepath);
  ~OIIOImageLoader();

  bool load_metadata(const ImageDeviceFeatures &features, ImageMetaData &metadata) override;

  bool load_pixels(const ImageMetaData &metadata,
                   void *pixels,
                   const size_t pixels_size,
                   const bool associate_alpha) override;

  string name() const override;

  ustring osl_filepath() const override;

  bool equals(const ImageLoader &other) const override;

  bool get_tx(const ustring &colorspace,
              const ExtensionType &extension,
              Progress *progress,
              bool auto_convert,
              const char *cache_path);

  static bool make_tx(const string &filename,
                      const string &outputfilename,
                      const ustring &colorspace,
                      ExtensionType extension);

 protected:
  ustring filepath;
};

CCL_NAMESPACE_END

#endif /* __IMAGE_OIIO__ */