using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using OpenCVMarkerLessAR;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.Calib3dModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.UnityUtils.Helper;
using HoloLensWithOpenCVForUnity.UnityUtils.Helper;

namespace OpenCVMarkerLessAR_Extension
{
    [Serializable]
    public class ReferenceImage
    {
        public Texture2D Texture;
        public float ImageSizeScale;
    }

    [RequireComponent(typeof(HololensCameraStreamToMatHelper))]
    [RequireComponent(typeof(ImageOptimizationHelper))]
    public class HoloLensImageDetector : MonoBehaviour
    {
        public List<ReferenceImage> ReferenceImageList = new List<ReferenceImage>();
        public Camera ARCamera;
        public GameObject ARObjectPrefab;

        bool _EnableDetection = true;
        public bool EnableDetection { get; }

        // The camera matrix value of Hololens camera 896x504 size.
        // For details on the camera matrix, please refer to this page. (http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html)
        // These values ​​are unique to my device, obtained from the "Windows.Media.Devices.Core.CameraIntrinsics" class. (https://docs.microsoft.com/en-us/uwp/api/windows.media.devices.core.cameraintrinsics)
        // Can get these values by using this helper script. (https://github.com/EnoxSoftware/HoloLensWithOpenCVForUnityExample/tree/master/Assets/HololensCameraIntrinsicsChecker/CameraIntrinsicsCheckerHelper)
        double _Fx = 1035.149;//focal length x.
        double _Fy = 1034.633;//focal length y.
        double _Cx = 404.9134;//principal point x.
        double _Cy = 236.2834;//principal point y.
        double _DistCoeffs1 = 0.2036923;//radial distortion coefficient k1.
        double _DistCoeffs2 = -0.2035773;//radial distortion coefficient k2.
        double _DistCoeffs3 = 0.0;//tangential distortion coefficient p1.
        double _DistCoeffs4 = 0.0;//tangential distortion coefficient p2.
        double _DistCoeffs5 = -0.2388065;//radial distortion coefficient k3.

        HololensCameraStreamToMatHelper _WebCamTextureToMatHelper;
        Mat _GrayMat;

        void Start()
        {
            _WebCamTextureToMatHelper = gameObject.GetComponent<HololensCameraStreamToMatHelper>();

#if WINDOWS_UWP && !DISABLE_HOLOLENSCAMSTREAM_API
            _WebCamTextureToMatHelper.frameMatAcquired += OnFrameMatAcquired;
#endif
            _WebCamTextureToMatHelper.onInitialized.AddListener(OnWebCamTextureToMatHelperInitialized);
            _WebCamTextureToMatHelper.onDisposed.AddListener(OnWebCamTextureToMatHelperDisposed);
            _WebCamTextureToMatHelper.onErrorOccurred.AddListener(OnWebCamTextureToMatHelperErrorOccurred);

            _WebCamTextureToMatHelper.Initialize();
        }

        void Update()
        {
            UpdateARObjectTransform();
        }

        void OnDestroy()
        {
#if WINDOWS_UWP && !DISABLE_HOLOLENSCAMSTREAM_API
            _WebCamTextureToMatHelper.frameMatAcquired -= OnFrameMatAcquired;
#endif
            _WebCamTextureToMatHelper.Dispose();
        }

        public void OnWebCamTextureToMatHelperInitialized()
        {
            _WebCamTextureToMatHelper.flipHorizontal = _WebCamTextureToMatHelper.GetWebCamDevice().isFrontFacing;
 
            Mat webCamTextureMat = _WebCamTextureToMatHelper.GetMat();
            _GrayMat = new Mat(webCamTextureMat.rows(), webCamTextureMat.cols(), CvType.CV_8UC1);

            MatOfDouble distCoeffs = new MatOfDouble(_DistCoeffs1, _DistCoeffs2, _DistCoeffs3, _DistCoeffs4, _DistCoeffs5);
            InitializeImageDetector(webCamTextureMat, _Fx, _Fy, _Cx, _Cy, distCoeffs);
        }

        public void OnWebCamTextureToMatHelperDisposed()
        {
            Debug.Log("OnWebCamTextureToMatHelperDisposed");

            if (_GrayMat != null)
            {
                _GrayMat.Dispose();
            }
        }

        public void OnWebCamTextureToMatHelperErrorOccurred(WebCamTextureToMatHelper.ErrorCode errorCode)
        {
            Debug.Log("OnWebCamTextureToMatHelperErrorOccurred " + errorCode);
        }

#if WINDOWS_UWP && !DISABLE_HOLOLENSCAMSTREAM_API
        public void OnFrameMatAcquired(Mat bgraMat, Matrix4x4 projectionMatrix, Matrix4x4 cameraToWorldMatrix)
        {
            if (_EnableDetection)
            {
                Imgproc.cvtColor(bgraMat, _GrayMat, Imgproc.COLOR_BGRA2GRAY);
                FindARMarker(_GrayMat, cameraToWorldMatrix);
            }
        }
#endif

        #region ImageDetector

        private Dictionary<string, Pattern> _Patterns = new Dictionary<string, Pattern>();
        private Dictionary<string, PatternDetector> _PatternDetectors = new Dictionary<string, PatternDetector>();
        private Dictionary<string, GameObject> _ARObjects = new Dictionary<string, GameObject>();
        private Dictionary<string, Matrix4x4> _ARObjectScaleMatrix = new Dictionary<string, Matrix4x4>();
        private Dictionary<string, Matrix4x4> _ARObjectTransformMatrix = new Dictionary<string, Matrix4x4>();
        private Dictionary<string, bool> _ARObjectHasUpdate = new Dictionary<string, bool>();

        /// <summary>
        /// The cameraparam matrix.
        /// </summary>
        Mat _CamMatrix;

        /// <summary>
        /// The dist coeffs.
        /// </summary>
        MatOfDouble _DistCoeffs;

        /// <summary>
        /// The matrix that inverts the Y axis.
        /// </summary>
        Matrix4x4 _InvertYMat;

        /// <summary>
        /// The matrix that inverts the Z axis.
        /// </summary>
        Matrix4x4 _InvertZMat;

        void InitializeImageDetector(Mat inputImageMat, double fx, double fy, double cx, double cy, MatOfDouble distCoeffs)
        {
            InitializePatternDetector();
            InitializeMatrix();
            InitializeCameraMatrix(inputImageMat, fx, fy, cx, cy, distCoeffs);
        }

        void InitializePatternDetector()
        {
            // Learning the feature points of the pattern image.
            foreach(ReferenceImage image in ReferenceImageList)
            {
                Texture2D patternTexture = image.Texture;

                Mat patternMat = new Mat(patternTexture.height, patternTexture.width, CvType.CV_8UC4);
                Utils.texture2DToMat(patternTexture, patternMat);

                Pattern pattern = new Pattern();
                PatternDetector patternDetector = new PatternDetector(null, null, null, true);

                patternDetector.buildPatternFromImage(patternMat, pattern);
                patternDetector.train(pattern);

                _Patterns[patternTexture.name] = pattern;
                _PatternDetectors[patternTexture.name] = patternDetector;
            }

            Debug.Log("**** _Patterns.Count @Initialize(): " + _Patterns.Count);
            Debug.Log("**** _PatternDetectors.Count @Initialize(): " + _PatternDetectors.Count);
        }

        void InitializeMatrix()
        {
            _InvertZMat = Matrix4x4.TRS (Vector3.zero, Quaternion.identity, new Vector3 (1, 1, -1));
            Debug.Log ("_InvertZMat " + _InvertZMat.ToString ());

            _InvertYMat = Matrix4x4.TRS (Vector3.zero, Quaternion.identity, new Vector3 (1, -1, 1));
            Debug.Log ("_InvertYMat " + _InvertYMat.ToString ());

            foreach(ReferenceImage image in ReferenceImageList)
            {
                float scale = image.ImageSizeScale;
                Matrix4x4 scaleMat = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3 (scale, scale, scale));
                _ARObjectScaleMatrix[image.Texture.name] = scaleMat;

                _ARObjectTransformMatrix[image.Texture.name] = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3 (1, 1, 1));
                _ARObjectHasUpdate[image.Texture.name] = false;
            }
        }

        void InitializeCameraMatrix(Mat inputImageMat, double fx, double fy, double cx, double cy, MatOfDouble distCoeffs)
        {
            Debug.Log ("******************************");

            float width = inputImageMat.width();
            float height = inputImageMat.height();

            // Set camera param
            _CamMatrix = new Mat (3, 3, CvType.CV_64FC1);
            _CamMatrix.put (0, 0, fx);
            _CamMatrix.put (0, 1, 0);
            _CamMatrix.put (0, 2, cx);
            _CamMatrix.put (1, 0, 0);
            _CamMatrix.put (1, 1, fy);
            _CamMatrix.put (1, 2, cy);
            _CamMatrix.put (2, 0, 0);
            _CamMatrix.put (2, 1, 0);
            _CamMatrix.put (2, 2, 1.0f);
            Debug.Log ("CamMatrix " + _CamMatrix.dump ());

            _DistCoeffs = distCoeffs;
            Debug.Log ("DistCoeffs " + _DistCoeffs.dump ());

            // Calibration camera
            Size imageSize = new Size (width, height);
            double apertureWidth = 0;
            double apertureHeight = 0;
            double[] fovx = new double[1];
            double[] fovy = new double[1];
            double[] focalLength = new double[1];
            Point principalPoint = new Point (0, 0);
            double[] aspectratio = new double[1];

            Calib3d.calibrationMatrixValues(_CamMatrix, imageSize, apertureWidth, apertureHeight, fovx, fovy, focalLength, principalPoint, aspectratio);

            Debug.Log ("ImageSize " + imageSize.ToString ());
            Debug.Log ("ApertureWidth " + apertureWidth);
            Debug.Log ("ApertureHeight " + apertureHeight);
            Debug.Log ("Fovx " + fovx [0]);
            Debug.Log ("Fovy " + fovy [0]);
            Debug.Log ("FocalLength " + focalLength [0]);
            Debug.Log ("PrincipalPoint " + principalPoint.ToString ());
            Debug.Log ("Aspectratio " + aspectratio [0]);

            Debug.Log ("******************************");
        }

        void FindARMarker(Mat imgMat, Matrix4x4 cameraToWorldMatrix)
        {
            PatternTrackingInfo patternTrackingInfo = new PatternTrackingInfo();
            foreach(string patternName in _Patterns.Keys)
            {
                bool patternFound = _PatternDetectors[patternName].findPattern(imgMat, patternTrackingInfo);
                // Debug.Log ("PatternFound " + patternFound);

                if(patternFound)
                {
                    patternTrackingInfo.computePose(_Patterns[patternName], _CamMatrix, _DistCoeffs);

                    Matrix4x4 transformationM = patternTrackingInfo.pose3d; // Marker to Camera Coordinate System Convert Matrix

                    Matrix4x4 scaleMat = _ARObjectScaleMatrix[patternName];

                    // _ARObjectTransformMatrix[patternName] = cameraToWorldMatrix * scaleMat * _InvertYMat * transformationM * _InvertZMat;
                    _ARObjectTransformMatrix[patternName] = cameraToWorldMatrix * scaleMat * _InvertZMat * _InvertYMat * transformationM * _InvertZMat;

                    _ARObjectHasUpdate[patternName] = true;
                }
                else
                {
                    _ARObjectHasUpdate[patternName] = false;
                }
            }
        }

        void UpdateARObjectTransform()
        {
            foreach(string patternName in _Patterns.Keys)
            {
                if(_ARObjectHasUpdate[patternName])
                {
                    _ARObjectHasUpdate[patternName] = false;

                    Matrix4x4 ARM = _ARObjectTransformMatrix[patternName];

                    GameObject ARGameObject;
                    if (!_ARObjects.TryGetValue(patternName, out ARGameObject))
                    {
                        ARGameObject = GameObject.Instantiate(ARObjectPrefab, Vector3.zero, Quaternion.identity);
                        ARGameObject.name = ARObjectPrefab.name + "_" + patternName;
                        _ARObjects[patternName] = ARGameObject;
                    }

                    ARUtils.SetTransformFromMatrix(ARGameObject.transform, ref ARM);
                }
            }
        }

        #endregion
    }
}
