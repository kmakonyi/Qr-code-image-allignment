#!/usr/bin/env python3
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from pyzbar import pyzbar
import glob
import os.path

def clean(limg, minA):
    cnts, _ = cv.findContours(limg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv.contourArea(cnt) < minA:
            cv.drawContours(limg, [cnt], 0, 0, -1)
    return limg


def cleanSlices(lslice):
    contours, _ = cv.findContours(lslice, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda x: cv.contourArea(x))
    if len(contours) > 0:
        while cv.boundingRect(contours[0])[3] < 5:
            del contours[0]
    contours = sorted(contours, key=lambda x: cv.boundingRect(x)[0])
    return contours


def fitplot(inimg, outimg, pos, neg, idx, direct, degree):

    ppp, slice = fitY(inimg, pos, neg, idx, degree)
    # print("|", str(ppp).split(' ')[1][1:], "|")
    (w, h) = inimg.shape
    x = list(range(h))
    y = ppp(x)
    maxshift = int(round(np.max(y)))
    aver = int(round(np.sum(y)/len(y)))
    if direct == 'Y':
        y = y-aver+pos[idx]
        img_c = np.copy(outimg)
        smallimg = img_c  #[:, maxshift:]

        # print('L97', h, outimg.shape)
        for xidx in range(h-1):
            shift = int(round(y[xidx]))
            if shift < outimg.shape[0]:
                # print(shift, xidx, outimg.shape, h, w)
                outimg[shift, xidx] = outimg[shift, xidx]+128


    if direct == 'X':
        xshift = w-pos[idx]
        y = 1*aver-y+xshift
#        y = xshift-aver-y
        # print(aver, idx, pos[idx], xshift)
        img_c = np.copy(outimg)
        smallimg = img_c  #[:, maxshift:]
        # print('L111', h, outimg.shape)
        for xidx in range(h-1):
            shift = int(round(y[xidx]))
            # print(xidx, shift)
            if shift < outimg.shape[1]:
                outimg[xidx, shift] = outimg[xidx, shift] + 128

    return slice

def findsep(projection, img):
    pos = []
    neg = []
    for idx, v in enumerate(projection):
        if idx != 0:
            if projection[idx-1] == 0 and projection[idx] > 0:
                pos.append(idx)
            if projection[idx-1] > 0 and projection[idx] == 0:
                neg.append(idx)

    IDX = len(pos)
    if len(neg) > len(pos):
        IDX = len(neg)

    for idx in range(IDX):
        roi = img[pos[idx]:neg[idx], :]
        M = cv.moments(roi)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        nonblackratio = cv.countNonZero(roi)/(roi.shape[0] * roi.shape[1])
        fnz = np.min(np.array(first_nonzero(roi, axis=1, invalid_val=-1)))
        lnz = np.max(np.array(last_nonzero(roi, axis=1, invalid_val=-1)))
        OK = False
        if (lnz-fnz)/roi.shape[1] > 0.7:
            OK = True
        if (lnz-fnz)/roi.shape[1] > 0.5:
            if nonblackratio > 0.1:
                if (0.4 < cX/roi.shape[1]) and (cX/roi.shape[1] < 0.6):
                    if (neg[idx] - pos[idx]) < 18:
                        OK = True
        # print('- - -', idx, neg[idx] - pos[idx], cX/roi.shape[1], nonblackratio, (lnz-fnz)/roi.shape[1], OK)
        # plt.subplot(1, 1, 1), plt.imshow(roi, cmap='gray')
        # plt.show()
        if idx > 0:
            avg = int((neg[idx-1]+pos[idx])/2)
        else:
            avg = int((pos[idx])/2)
         # cv.line(img, (0, avg), (h, avg), 255, 1)
            # print(idx, avg)
    return pos, neg

#######


def normalize_image(limg):

    imgmin = np.min(limg)
    imgmax = np.max(limg)
    scalefact = 255/(imgmax-imgmin)
    bimg = (limg-imgmin)*scalefact
    bimg = bimg.astype(np.uint8)
    # print(np.min(limg), np.max(limg), np.min(bimg), np.max(bimg))
    return bimg


# def prepraration(img, dthr):

#     img normalize_image(img)

#     sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=15)
#     sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=15)

#     min = np.min(sobelx)
#     max = np.max(sobelx)
#     diff = max-min
#     mult = 255/diff
#     sobelxi = mult*(sobelx-min)
#     sobelxi = sobelxi.astype(np.uint8)
#     min = np.min(sobely)
#     max = np.max(sobely)
#     diff = max-min
#     mult = 255/diff
#     sobelyi = mult*(sobely-min)
#     sobelyi = sobelyi.astype(np.uint8)

#     #dthr = 50
#     hist = cv.calcHist([sobelxi], [0], None, [256], [0,256])
#     max = np.where(hist == np.max(hist))[0][0]
#     ret, thsobelxh = cv.threshold(sobelxi, max+dthr, 255, cv.THRESH_TOZERO)

#     hist = cv.calcHist([sobelxi], [0], None, [256], [0,256])
#     max = np.where(hist == np.max(hist))[0][0]
#     ret, thsobelxl = cv.threshold((255-sobelxi), max+dthr, 255, cv.THRESH_TOZERO)

#     plt.subplot(2, 2, 1), plt.imshow(sobelxi, cmap='gray')
#     plt.subplot(2, 2, 2), plt.imshow(sobelyi, cmap='gray')
#     plt.subplot(2, 2, 3), plt.plot(hist, color='gray')
#     plt.show()

#     hist = cv.calcHist([sobelxi], [0], None, [256], [0,256])
#     max = np.where(hist == np.max(hist))[0][0]
#     ret, thsobelyh = cv.threshold(sobelyi, max+dthr, 255, cv.THRESH_TOZERO)

#     hist = cv.calcHist([sobelxi], [0], None, [256], [0,256])
#     max = np.where(hist == np.max(hist))[0][0]
#     ret, thsobelyl = cv.threshold((255-sobelyi), max+dthr, 255, cv.THRESH_TOZERO)



#     thsobelxh = clean(thsobelxh, 225)
#     thsobelxl = clean(thsobelxl, 225)
#     thsobelyh = clean(thsobelyh, 225)
#     thsobelyl = clean(thsobelyl, 225)

#     (h, w) = img.shape[:2]
#     # # calculate the center of the image
#     center = (w / 2, h / 2)

#     M = cv.getRotationMatrix2D(center, 90, 1)
#     thsobelxh = cv.warpAffine(thsobelxh, M, (h, w))
#     thsobelxl = cv.warpAffine(thsobelxl, M, (h, w))

#     p_sobelxh = np.sum(thsobelxh, axis=1)
#     p_sobelxl = np.sum(thsobelxl, axis=1)
#     p_sobelyh = np.sum(thsobelyh, axis=1)
#     p_sobelyl = np.sum(thsobelyl, axis=1)

#     posxh, negxh = findsep(p_sobelxh, thsobelxh)
#     print("------------------------------")
#     posxl, negxl = findsep(p_sobelxl, thsobelxl)
#     print("------------------------------")
#     posyh, negyh = findsep(p_sobelyh, thsobelyh)
#     print("------------------------------")
#     posyl, negyl = findsep(p_sobelyl, thsobelyl)

#     return thsobelxl, posxl, negxl, thsobelxh, posxh, negxh,  thsobelyl, posyl, negyl, thsobelyh, posyh, negyh


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


#def findboundaries(limg, axes, inv=False, rotate=False, first=True):
def findboundaries(limg, inv=False, rotate=False):
    img_norm = normalize_image(limg)
    if inv is True:
        img_norm = 255-img_norm
    if rotate == False:
        sobel = cv.Sobel(img_norm, cv.CV_64F, 0, 1, ksize=15)
    else:
        sobel = cv.Sobel(img_norm, cv.CV_64F, 1, 0, ksize=15)

    sobeli = normalize_image(sobel)

    if rotate is True:
        (w, h) = sobeli.shape
        center = (w / 2, h / 2)
        M = cv.getRotationMatrix2D(center, -90, 1)
        sobeli = cv.warpAffine(sobeli, M, (w, h))
        sobeli = cv.flip(sobeli, 1)

    hist = cv.calcHist([sobeli], [0], None, [256], [0,256])
    max = np.where(hist == np.max(hist))[0][0]
    _, th = cv.threshold(sobeli, max+50, 255, cv.THRESH_TOZERO)

    pth = np.sum(th, axis=1)
    pos, neg = findsep(pth, th)
    bool = isroifine(th, pos, neg)
    return th, pos, neg, bool


def isroifine(limg, lpos, lneg):
    IDX = len(lpos)
    if len(lneg) > len(lpos):
        IDX = len(lneg)

    bvec = []

    for idx in range(IDX):
        roi = limg[lpos[idx]:lneg[idx], :]
        M = cv.moments(roi)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        nonblackratio = cv.countNonZero(roi)/(roi.shape[0] * roi.shape[1])
        fnz = np.min(np.array(first_nonzero(roi, axis=1, invalid_val=-1)))
        lnz = np.max(np.array(last_nonzero(roi, axis=1, invalid_val=-1)))
        contours, hierarchy = cv.findContours(roi, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        Nblob = len(contours)
        height = lneg[idx] - lpos[idx]
        OK = False
        if Nblob < 6:
            if height < 17:
                if height < 17:
                    if (lnz-fnz)/roi.shape[1] > 0.7:
                        OK = True
                    if (lnz-fnz)/roi.shape[1] > 0.5:
                        if nonblackratio > 0.1:
                            if (0.4 < cX/roi.shape[1]) and (cX/roi.shape[1] < 0.6):
                                OK = True

        # print('- - -', idx, height, cX/roi.shape[1], nonblackratio, (lnz-fnz)/roi.shape[1], len(contours), OK)
#        if OK:
        # plt.subplot(1, 1, 1), plt.imshow(roi, cmap='gray')
        # plt.show()
        bvec.append(OK)

    return bvec

def average(limg):
    _lw = np.arange(1, limg.shape[0])
    x = []
    y = []

    for idx in range((limg.shape[1])):
        lslice = limg[:, idx]
        slslice = np.sum(lslice)
        aver = 0
        for i, v in enumerate(lslice):
            aver = aver + (_lw[i-1]*v)
            # print(idx, i, aver)

        aver2 = round(aver/slslice)

        if not math.isnan(aver2):
            x.append(idx)
            y.append(aver/slslice)


    # aver = np.average(limg, axis=axis)
    # sumv = np.sum(limg, axis=axis)
    # for idx in range(len(aver)):
    #     print(idx, aver[idx], sumv[idx], sumv[idx]/aver[idx])

    # plt.subplot(1, 1, 1), plt.imshow(limg, cmap='gray'), plt.scatter(x, y)
    # plt.show()
    return x, y


def _fitY(lx, ly, shift, fullwidth):

    xx = np.array(lx).reshape((-1, 1))
    yy = np.array(ly).reshape((-1, 1))

    ransac = linear_model.RANSACRegressor(residual_threshold=4)
    model = make_pipeline(PolynomialFeatures(2), ransac)
    model.fit(xx, ly)

    # ------
    x = np.arange(fullwidth).reshape(-1, 1)
    # ------
    line_y_ransac = model.predict(x)+shift

    return x, line_y_ransac


def fitY(limg, lpos, lneg, lbool, lidx, tmpimage, rot):

    if lbool[lidx] is False:
        return None, None

    roi = limg[lpos[lidx]:lneg[lidx], :]
    x, y = average(roi)
    # print(len(x), len(y), type(x), type(y), roi.shape)
    xx, yy = _fitY(x, y, lpos[lidx], roi.shape[1])

    x = xx.flatten()
    # print(x.shape, line_y_ransac.shape)

    for idx in range(x.shape[0]):
        idxy = int(round(yy[idx]))
        # print(idx, x[idx], idxy)
        if rot is False:
            tmpimage[x[idx], idxy] = tmpimage[x[idx], idxy] + 128
        else:
            tmpimage[idxy, x[idx]] = tmpimage[idxy, x[idx]] + 128

    # plt.subplot(1, 1, 1), plt.imshow(limg, cmap='gray'), plt.scatter(xx, yy)
    # plt.show()
    # print("end of fit")
    return x, yy


def firstfit(limg, first, tmpimage, rot):
    if first is True:
        fnz = np.array(first_nonzero(limg, axis=0, invalid_val=-1))
    if first is False:
        fnz = np.array(last_nonzero(limg, axis=0, invalid_val=-1))

    x = np.arange(len(fnz))
    xx, yy = _fitY(x, fnz, 0, limg.shape[1])

    x = xx.flatten()
    # print(x.shape, line_y_ransac.shape)

    for idx in range(x.shape[0]):
        idxy = int(round(yy[idx]))
        # print(idx, x[idx], idxy)
        if rot is False:
            tmpimage[idxy, x[idx]] = tmpimage[idxy, x[idx]] + 128
        else:
            tmpimage[x[idx], idxy] = tmpimage[x[idx], idxy] + 128

    # plt.subplot(1, 1, 1), plt.imshow(limg, cmap='gray'), plt.scatter(xx, yy)
    # plt.show()
    return x, yy

''' ---------------------------------- '''

def process(fname):
    img = cv.imread('00.bmp', 0)
    (w, h) = img.shape
    img = cv.resize(img, (4*w, 4*h), interpolation=cv.INTER_AREA)
    (w, h) = img.shape
    img = cv.GaussianBlur(img,(5,5),0)
    imgnorm = normalize_image(img)
    cv.imwrite("n00.bmp", imgnorm)

    thimg1, pos1, neg1, bool1 = findboundaries(img, inv=True, rotate=False)
    thimg2, pos2, neg2, bool2 = findboundaries(img, inv=False, rotate=False)
    # print("----")
    thimg3, pos3, neg3, bool3 = findboundaries(img, inv=True, rotate=True)
    thimg4, pos4, neg4, bool4 = findboundaries(img, inv=False, rotate=True)

    maxidx = np.max(img.shape)
    # print('max:', maxidx)
    template0 = np.zeros((maxidx, maxidx))
    xx1, yy1 = firstfit(thimg1, True, template0, False)
    xx2, yy2 = firstfit(thimg2, False, template0, False)
    xx3, yy3 = firstfit(thimg3, True, template0, True)
    xx4, yy4 = firstfit(thimg4, False, template0, True)

    points = np.where(template0 > 129)
    # print(points)
    # ################ 2, find the 4 corners to rectify
    template = np.zeros((maxidx, maxidx))
    shift = 15
    ptr_src = []
    ptr_dst = []
    for idx in range(len(points[0])):
        ptr_src.append((points[1][idx], points[0][idx]))
        ptr_dst.append((15, 15))
        ptr_dst.append((15+(template.shape[0]-2*shift), 15))
        ptr_dst.append((15+(template.shape[0]-2*shift), 15+(template.shape[0]-2*shift)))
        ptr_dst.append((15, 15+(template.shape[0]-2*shift)))

    template[0:img.shape[0], 0:img.shape[1]] = img
    template_dst = np.zeros(template.shape)
    #template_dst = cv.remap(template, template_x, template_y, cv.INTER_LINEAR)

    sourceshape = np.array(ptr_src,np.int32)
    sourceshape=sourceshape.reshape(1,-1,2)
    targetshape = np.array(ptr_dst,np.int32)
    targetshape=targetshape.reshape(1,-1,2)
    # print(sourceshape, targetshape)
    matches =[]
    for idx in range(len(ptr_src)):
        # print(idx, ptr_src[idx], ptr_dst[idx], template.shape)
        matches.append(cv.DMatch(idx, idx, 0)) #

    # ################ 3, rectify (part1)
    tps = cv.createThinPlateSplineShapeTransformer()
    tps.estimateTransformation(targetshape ,sourceshape, matches)
    tps.warpImage(template, template_dst, cv.INTER_CUBIC, cv.BORDER_REPLICATE)

    barcode1 = pyzbar.decode(imgnorm)
    barcode2 = pyzbar.decode(template_dst)

    barcodeData1 = ""
    barcodeData2 = ""
    for barcode in barcode1:
	    # # extract the bounding box location of the barcode and draw the
	    # # bounding box surrounding the barcode on the image
	    # (x, y, w, h) = barcode.rect
	    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
	    # the barcode data is a bytes object so if we want to draw it on
	    # our output image we need to convert it to a string first
	    barcodeData1 = barcode.data.decode("utf-8")
	    barcodeType = barcode.type
	    # # draw the barcode data and barcode type on the image
	    # text = "{} ({})".format(barcodeData, barcodeType)
	    # cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
		#             0.5, (0, 0, 255), 2)
	    # print the barcode type and data to the terminal
        # text1 = "Before: {}".format(barcodeData)
	    # print("[INFO] File {} Found {} barcode: {}".format(fname, barcodeType, barcodeData))


    for barcode in barcode2:
	    barcodeData2 = barcode.data.decode("utf-8")
	    barcodeType = barcode.type
        #text2 = barcodeData
	    # print("[INFO] File {} Found {} barcode: {}".format(fname, barcodeType, barcodeData))
    fnameo = fname.split('.')[0]+"_d.bmp"
    print(fname, fnameo, "------\"", barcodeData1, "\" -> \"", barcodeData2, "\"")
    cv.imwrite(fnameo, template_dst)
    diff = template.shape[1]-img.shape[1]
    # plt.subplot(2, 2, 1), plt.imshow(template, cmap='gray'), \
    #     plt.scatter(xx1, yy1, marker='.'), \
    #     plt.scatter(xx2, yy2, marker='.'), \
    #     plt.scatter(yy3+diff/2, xx3, marker='.'), \
    #     plt.scatter(yy4+diff/2, xx4, marker='.'), \
    #     plt.scatter(ptr_src[0][0]+diff/2, ptr_src[0][1], marker='o'), \
    #     plt.scatter(ptr_src[1][0]+diff/2, ptr_src[1][1], marker='o'), \
    #     plt.scatter(ptr_src[2][0]+diff/2, ptr_src[2][1], marker='o'), \
    #     plt.scatter(ptr_src[3][0]+diff/2, ptr_src[3][1], marker='o'), \
    #     plt.scatter(ptr_dst[0][0], ptr_dst[0][1], marker='.'), \
    #     plt.scatter(ptr_dst[1][0], ptr_dst[1][1], marker='.'), \
    #     plt.scatter(ptr_dst[2][0], ptr_dst[2][1], marker='.'), \
    #     plt.scatter(ptr_dst[3][0], ptr_dst[3][1], marker='.'), \
    # plt.subplot(2, 2, 2), plt.imshow(template_dst, cmap='gray')
    # plt.subplot(2, 2, 3), plt.imshow(template0, cmap='gray')
    # plt.show()



files = glob.glob("pic_*.bmp")
# files = []
# for i in range(16):
#     files.append("pic_16.bmp")
for file in files:
    process(file)

'''



thimg1, pos1, neg1, bool1 = findboundaries(template_dst, inv=True, rotate=False)
thimg2, pos2, neg2, bool2 = findboundaries(template_dst, inv=False, rotate=False)
print("----")
thimg3, pos3, neg3, bool3 = findboundaries(template_dst, inv=True, rotate=True)
thimg4, pos4, neg4, bool4 = findboundaries(template_dst, inv=False, rotate=True)

# maxidx = np.max(img.shape)
# print('max:', maxidx)
X1 = []
Y1 = []
X2 = []
Y2 = []
X3 = []
Y3 = []
X4 = []
Y4 = []
template = np.zeros((maxidx, maxidx))
for idx in range(len(pos1)):
    tx, ty = fitY(thimg1, pos1, neg1, bool1, idx, template, rot=False)
    if tx is not None:
        X1.append(tx)
        Y1.append(ty)

for idx in range(len(pos3)):
    tx, ty = fitY(thimg3, pos3, neg3, bool3, idx, template, rot=True)
    if tx is not None:
        X3.append(tx)
        Y3.append(ty)

for idx in range(len(pos2)):
    tx, ty = fitY(thimg2, pos2, neg2, bool2, idx, template, rot=False)
    if tx is not None:
        X2.append(tx)
        Y2.append(ty)

for idx in range(len(pos4)):
    tx, ty = fitY(thimg4, pos4, neg4, bool4, idx, template, rot=True)
    if tx is not None:
        X4.append(tx)
        Y4.append(ty)

plt.subplot(1, 1, 1), plt.imshow(template, cmap='gray')
plt.show()
plt.subplot(1, 1, 1), plt.imshow(template_dst, cmap='gray')
for idx in range(len(X1)):
    plt.scatter(X1[idx], Y1[idx], marker='.')
for idx in range(len(X2)):
    plt.scatter(X2[idx], Y2[idx], marker='.')
for idx in range(len(X3)):
    plt.scatter(Y3[idx], X3[idx], marker='.')
for idx in range(len(X4)):
    plt.scatter(Y4[idx], X4[idx], marker='.')
plt.show()
'''
