
#include <cuml/svm/svm_api.h>

void test_svm() {

   cumlHandle_t handle = 0;
   cumlError_t response = CUML_SUCCESS;

   response = cumlSpSvcFit(handle, NULL, 0, 1, NULL, 1.0f, 2.0f, 2, 3, 3.0f, 4, LINEAR, 5, 6.0f, 7.0f, NULL, NULL, NULL, NULL, NULL, NULL, NULL);

   response = cumlDpSvcFit(handle, NULL, 0, 1, NULL, 1.0, 2.0, 2, 3, 3.0, 4, LINEAR, 5, 6.0, 7.0, NULL, NULL, NULL, NULL, NULL, NULL, NULL);

   response = cumlSpSvcPredict(handle, NULL, 0, 1, LINEAR, 2, 3.0f, 4.0f, 5, 6.0f, NULL, NULL, 7, NULL, NULL, 8.0f, 9);

   response = cumlDpSvcPredict(handle, NULL, 0, 1, LINEAR, 2, 3.0, 4.0, 5, 6.0, NULL, NULL, 7, NULL, NULL, 8.0, 9);
}