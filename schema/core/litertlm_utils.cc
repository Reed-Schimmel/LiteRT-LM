#include "schema/core/litertlm_utils.h"

#include <string>

#include "schema/core/litertlm_header_schema_generated.h"

namespace litert {
namespace litertlm {
namespace schema {

std::string AnySectionDataTypeToString(AnySectionDataType value) {
  switch (value) {
    case AnySectionDataType_NONE:
      return "AnySectionDataType_NONE";
    case AnySectionDataType_Deprecated:
      return "AnySectionDataType_Deprecated";
    case AnySectionDataType_TFLiteModel:
      return "AnySectionDataType_TFLiteModel";
    case AnySectionDataType_SP_Tokenizer:
      return "AnySectionDataType_SP_Tokenizer";
    case AnySectionDataType_LlmMetadataProto:
      return "AnySectionDataType_LlmMetadataProto";
    case AnySectionDataType_GenericBinaryData:
      return "AnySectionDataType_GenericBinaryData";
    default:
      // Handle cases for MIN/MAX or potentially invalid values.
      return "Unknown AnySectionDataType value";
  }
}

}  // namespace schema
}  // namespace litertlm
}  // namespace litert
