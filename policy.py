#Policy found on CIFAR-10 and CIFAR-100
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def good_policies():
  exp0_0 = [
      [('Solarize', 0.66, 0.34), ('Equalize', 0.56, 0.61)],
      [('Equalize', 0.43, 0.06), ('AutoContrast', 0.66, 0.08)],
      [('Color', 0.72, 0.47), ('Contrast', 0.88, 0.86)],
      [('Brightness', 0.84, 0.71), ('Color', 0.31, 0.74)],
      [('Rotate', 0.68, 0.26), ('TranlateX', 0.38, 0.88)]]
  exp0_1 = [
      [('TranslateY', 0.88, 0.96), ('TranslateY', 0.53, 0.79)],
      [('AutoContrast', 0.44, 0.36), ('Solarize', 0.22, 0.48)],
      [('AutoContrast', 0.93, 0.32), ('Solarize', 0.85, 0.26)],
      [('Solarize', 0.55, 0.38), ('Equalize', 0.43, 0.48)],
      [('TranslateY', 0.72, 0.93), ('AutoContrast', 0.83, 0.95)]]
  exp0_2 = [
      [('Solarize', 0.43, 0.58), ('AutoContrast', 0.82, 0.26)],
      [('TranslateY', 0.71, 0.79), ('AutoContrast', 0.81, 0.94)],
      [('AutoContrast', 0.92, 0.18), ('TranslateY', 0.77, 0.85)],
      [('Equalize', 0.71, 0.69), ('Color', 0.23, 0.33)],
      [('Sharpness', 0.36, 0.98), ('Brightness', 0.72, 0.78)]]
  exp0_3 = [
      [('Equalize', 0.74, 0.49), ('TranslateY', 0.86, 0.91)],
      [('TranslateY', 0.82, 0.91), ('TranslateY', 0.96, 0.79)],
      [('AutoContrast', 0.53, 0.37), ('Solarize', 0.39, 0.47)],
      [('TranslateY', 0.22, 0.78), ('Color', 0.91, 0.65)],
      [('Brightness', 0.82, 0.46), ('Color', 0.23, 0.91)]]
  exp0_4 = [
      [('Cutout', 0.27, 0.45), ('Equalize', 0.37, 0.21)],
      [('Color', 0.43, 0.23), ('Brightness', 0.65, 0.71)],
      [('ShearX', 0.49, 0.31), ('AutoContrast', 0.92, 0.28)],
      [('Equalize', 0.62, 0.59), ('Equalize', 0.38, 0.91)],
      [('Solarize', 0.57, 0.31), ('Equalize', 0.61, 0.51)]]
  exp0_5 = [
      [('TranslateY', 0.29, 0.35), ('Sharpness', 0.31, 0.64)],
      [('Color', 0.73, 0.77), ('TranlateX', 0.65, 0.76)],
      [('ShearY', 0.29, 0.74), ('Posterize', 0.42, 0.58)],
      [('Color', 0.92, 0.79), ('Equalize', 0.68, 0.54)],
      [('Sharpness', 0.87, 0.91), ('Sharpness', 0.93, 0.41)]]
  exp0_6 = [
      [('Solarize', 0.39, 0.35), ('Color', 0.31, 0.44)],
      [('Color', 0.33, 0.77), ('Color', 0.25, 0.46)],
      [('ShearY', 0.29, 0.74), ('Posterize', 0.42, 0.58)],
      [('AutoContrast', 0.32, 0.79), ('Cutout', 0.68, 0.34)],
      [('AutoContrast', 0.67, 0.91), ('AutoContrast', 0.73, 0.83)]]
 
  return  exp0_0 + exp0_1 + exp0_2 + exp0_3 + exp0_4 + exp0_5 + exp0_6