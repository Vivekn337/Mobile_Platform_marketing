
/**************************Part 1*********************************/
data mobile;
set "DATA.sas7bdat";
run;

data lin_prob;
set mobile;
if device_platform_class ='android' then dummy_platform_class=0;
else dummy_platform_class=1;
run;

proc surveyselect data=mobile out=mobile_sampled outall samprate=0.8 seed=10;
run;

data mobile_training mobile_test;
 set mobile_sampled;
 if selected then output mobile_training; 
 else output mobile_test;
run;

/*Final model */ 
proc glmselect data=mobile_training testdata=mobile_test seed=2 plots=ase;
 class publisher_id_class(split) device_make_class(split) device_platform_class(split) device_os_class(split);
 model install = device_volume|wifi|resolution|device_height|device_width|publisher_id_class|device_os_class|device_make_class|device_platform_class @2
  /selection=stepwise(select=cv sle=0.1) hierarchy=single cvmethod=random(10) stats=all showpvalues;
 performance buildsscp=incremental;
run;/* Continues variables like resolution device height device width and device volume has been approximately normally distributed*/

/* Proof for not using rare events */
data rare_events;
set mobile;
if install=1 then output;
run;

/*Creating indicator variables for logistic model*/

proc glmmod data=mobile_training outdesign=mobile_tran_with_indicators noprint; 
 class publisher_id_class device_make_class device_platform_class device_os_class;
 model  install = device_volume wifi resolution device_height device_width publisher_id_class device_os_class device_make_class device_platform_class/ noint;
run;

proc glmmod data=mobile_test outdesign=mobile_test_with_indicators noprint; 
 class publisher_id_class device_make_class device_platform_class device_os_class;
 model  install = device_volume wifi resolution device_height device_width publisher_id_class device_os_class device_make_class device_platform_class/ noint;
run;

data mobile_test_with_indicators;
rename Col1=device_volume Col2=wifi Col3=resolution Col4=device_height Col5=device_width Col6=publisher_id_class1 Col7=publisher_id_class2 Col8=publisher_id_class3 Col9=publisher_id_class4 Col10=publisher_id_class5 Col11=publisher_id_class6 Col12=publisher_id_class7 Col13=publisher_id_class8 Col14=publisher_id_class9 Col15=publisher_id_class10 Col16=device_os_class1 Col17=device_os_class2 Col18=device_os_class3 Col19=device_os_class4 Col20=device_os_class5 Col21=device_os_class6 Col22=device_os_class7 Col23=device_os_class8 Col24=device_os_class9 
Col25=device_os_class10 Col26=device_make_class1 Col27=device_make_class2 Col28=device_make_class3 Col29=device_make_class4 Col30=device_make_class5 Col31=device_make_class6 Col32=device_make_class7 Col33=device_make_class8 Col34=device_make_class9 Col35=device_make_class10 Col36=device_platform_clas_android col37=device_platform_clas_iOS;
set mobile_test_with_indicators;
run;

data mobile_tran_with_indicators;
rename Col1=device_volume Col2=wifi Col3=resolution Col4=device_height Col5=device_width Col6=publisher_id_class1 Col7=publisher_id_class2 Col8=publisher_id_class3 Col9=publisher_id_class4 Col10=publisher_id_class5 Col11=publisher_id_class6 Col12=publisher_id_class7 Col13=publisher_id_class8 Col14=publisher_id_class9 Col15=publisher_id_class10 Col16=device_os_class1 Col17=device_os_class2 Col18=device_os_class3 Col19=device_os_class4 Col20=device_os_class5 Col21=device_os_class6 Col22=device_os_class7 Col23=device_os_class8 Col24=device_os_class9 
Col25=device_os_class10 Col26=device_make_class1 Col27=device_make_class2 Col28=device_make_class3 Col29=device_make_class4 Col30=device_make_class5 Col31=device_make_class6 Col32=device_make_class7 Col33=device_make_class8 Col34=device_make_class9 Col35=device_make_class10 Col36=device_platform_clas_android col37=device_platform_clas_iOS;
set mobile_tran_with_indicators;
run;

proc logistic data=mobile_tran_with_indicators outest=betas covout;
  model  install(event='1') = device_volume|wifi|resolution|device_height|device_width|publisher_id_class1 |publisher_id_class2| publisher_id_class3 |publisher_id_class4 |publisher_id_class5| publisher_id_class6| publisher_id_class7 |publisher_id_class8| publisher_id_class9| publisher_id_class10| device_os_class1| device_os_class2 |device_os_class3| device_os_class4| device_os_class5 |device_os_class6| device_os_class7| device_os_class8 |device_os_class9 |
device_os_class10 |device_make_class1 |device_make_class2 |device_make_class3| device_make_class4 |device_make_class5| device_make_class6| device_make_class7| device_make_class8 |device_make_class9| device_make_class10 |device_platform_clas_android |device_platform_clas_iOS  
 @2/ selection=stepwise slentry=0.1  slstay=0.1;
   output out=pred p=phat lower=lcl upper=ucl
          predprob=(individual crossvalidate);
   ods output Association=Association;
run;

model install (event='1')= wifi device_height publisher_id_class1 publisher_id_class2 publisher_id_class3 wifi*publisher_id_class3 publisher_id_class5 publisher_id_class6 publisher_id_class7 wifi*publisher_id_class7 device_height*publisher_id_class7 publisher_id_class8 device_os_class3 publisher_id_class7*device_os_class3 device_os_class9 device_make_class7 
model install(event='1')= wifi device_height publisher_id_class2 publisher_id_class3 publisher_id_class5 publisher_id_class7 publisher_id_class8 wifi*publisher_id_class7 device_height*publisher_id_class2 device_height*publisher_id_class5 device_height*publisher_id_class7 device_os_class3 device_os_class9 publisher_id_class7*device_os_class3 device_make_class7 device_make_class10 wifi*device_make_class7 publisher_id_class2*device_make_class7 publisher_id_class7*device_make_class7;;

/* Logit model without oversampling */
proc logistic data=mobile_tran_with_indicators;
 logit:  model install (event='1')= wifi device_height publisher_id_class1 publisher_id_class2 publisher_id_class3 wifi*publisher_id_class3 publisher_id_class5 publisher_id_class6 publisher_id_class7 wifi*publisher_id_class7 device_height*publisher_id_class7 publisher_id_class8 device_os_class3 publisher_id_class7*device_os_class3 device_os_class9 device_make_class7; 
  score data=mobile_test_with_indicators out=mobile_logit_predict;
run;

/*Oversampling */

proc logistic data=over_sample_tran outest=over_betas covout;
  model  install(event='1') = device_volume|wifi|resolution|device_height|device_width|publisher_id_class1 |publisher_id_class2| publisher_id_class3 |publisher_id_class4 |publisher_id_class5| publisher_id_class6| publisher_id_class7 |publisher_id_class8| publisher_id_class9| publisher_id_class10| device_os_class1| device_os_class2 |device_os_class3| device_os_class4| device_os_class5 |device_os_class6| device_os_class7| device_os_class8 |device_os_class9 |
device_os_class10 |device_make_class1 |device_make_class2 |device_make_class3| device_make_class4 |device_make_class5| device_make_class6| device_make_class7| device_make_class8 |device_make_class9| device_make_class10 |device_platform_clas_android |device_platform_clas_iOS  
 @2/ selection=stepwise slentry=0.1  slstay=0.1;
   output out=pred p=phat lower=lcl upper=ucl
          predprob=(individual crossvalidate);
   ods output Association=Association;
run;

model install (event='1')= wifi resolution publisher_id_class2 resolution*publisher_id_class2 publisher_id_class3 publisher_id_class6 publisher_id_class10 device_os_class1  publisher_id_class6*device_os_class1 device_make_class3 device_make_class7 wifi*device_make_class7 device_make_class8 device_make_class10 device_os_class1*device_make_class10  

proc glmmod data=mobile outdesign=mobile_main noprint; 
 class publisher_id_class device_make_class device_platform_class device_os_class;
 model  install = device_volume wifi resolution device_height device_width publisher_id_class device_os_class device_make_class device_platform_class/ noint;
run;

data mobile_main;
rename Col1=device_volume Col2=wifi Col3=resolution Col4=device_height Col5=device_width Col6=publisher_id_class1 Col7=publisher_id_class2 Col8=publisher_id_class3 Col9=publisher_id_class4 Col10=publisher_id_class5 Col11=publisher_id_class6 Col12=publisher_id_class7 Col13=publisher_id_class8 Col14=publisher_id_class9 Col15=publisher_id_class10 Col16=device_os_class1 Col17=device_os_class2 Col18=device_os_class3 Col19=device_os_class4 Col20=device_os_class5 Col21=device_os_class6 Col22=device_os_class7 Col23=device_os_class8 Col24=device_os_class9 
Col25=device_os_class10 Col26=device_make_class1 Col27=device_make_class2 Col28=device_make_class3 Col29=device_make_class4 Col30=device_make_class5 Col31=device_make_class6 Col32=device_make_class7 Col33=device_make_class8 Col34=device_make_class9 Col35=device_make_class10 Col36=device_platform_clas_android col37=device_platform_clas_iOS;
set mobile_main;
run;
 
data over_sample_tran;
set mobile_main;
if install=1 or (install=0 and ranuni(75302)< 1/121) then output;
run;

proc freq data=over_sample_tran;
table install;
run;/*r1=0.48955*/

proc freq data=mobile_main;
table install;
run; /*P1=0.0083*/

data over_sample_tran;
set over_sample_tran;
r1=0.48955;
p1=0.0083;
if install=1 then w=(p1/r1);
else w=(1-p1)/(1-r1);
off= log( (r1*(1-p1)) / ((1-r1)*p1) );
run;
/*Model with oversampling data */
/*Using weights*/
proc logistic data=over_sample_tran;
 /*logit: model install (event='1')= wifi device_height publisher_id_class1 publisher_id_class2 publisher_id_class3 wifi*publisher_id_class3 publisher_id_class5 publisher_id_class6 publisher_id_class7 wifi*publisher_id_class7 device_height*publisher_id_class7 publisher_id_class8 device_os_class3 publisher_id_class7*device_os_class3 device_os_class9 device_make_class7; */
logit : model install (event='1')= wifi resolution publisher_id_class2 resolution*publisher_id_class2 publisher_id_class3 publisher_id_class6 publisher_id_class10 device_os_class1  publisher_id_class6*device_os_class1 device_make_class3 device_make_class7 wifi*device_make_class7 device_make_class8 device_make_class10 device_os_class1*device_make_class10 ; 
weight w;/* score data=mobile_test_with_indicators out=mobile_logit_predict;*/
output out=out p=pwt;
run;

/*using offset*/
proc logistic data=over_sample_tran;
 /*logit: model install (event='1')= wifi device_height publisher_id_class1 publisher_id_class2 publisher_id_class3 wifi*publisher_id_class3 publisher_id_class5 publisher_id_class6 publisher_id_class7 wifi*publisher_id_class7 device_height*publisher_id_class7 publisher_id_class8 device_os_class3 publisher_id_class7*device_os_class3 device_os_class9 device_make_class7*/ 
logit : model install (event='1')= wifi resolution publisher_id_class2 resolution*publisher_id_class2 publisher_id_class3 publisher_id_class6 publisher_id_class10 device_os_class1  publisher_id_class6*device_os_class1 device_make_class3 device_make_class7 wifi*device_make_class7 device_make_class8 device_make_class10 device_os_class1*device_make_class10  
/ offset=off;/* score data=mobile_test_with_indicators out=mobile_logit_predict;*/
run;

/*ROC curves for the models*/
data mobile_main;
set mobile_main;
wifi_pic7=wifi*publisher_id_class7;
dh_pic2=device_height*publisher_id_class2;
dh_pic5=device_height*publisher_id_class5; 
dh_pic7=device_height*publisher_id_class7; 
pic7_doc3=publisher_id_class7*device_os_class3;  
wifi_dmc7=wifi*device_make_class7; 
pic2_dmc7=publisher_id_class2*device_make_class7; 
pic7_dmc7=publisher_id_class7*device_make_class7;
run;
install (event='1')= wifi device_height publisher_id_class2 publisher_id_class3 publisher_id_class5 publisher_id_class7 publisher_id_class8 wifi_pic7 dh_pic2 dh_pic5 dh_pic7 device_os_class3 device_os_class9 pic7_doc3 device_make_class7 device_make_class10 wifi_dmc7 pic2_dmc7 pic7_dmc7;

proc reg data=mobile_main;
 linear: model install= wifi device_height publisher_id_class2 publisher_id_class3 publisher_id_class5 publisher_id_class7 publisher_id_class8 wifi_pic7 dh_pic2 dh_pic5 dh_pic7 device_os_class3 device_os_class9 pic7_doc3 device_make_class7 device_make_class10 wifi_dmc7 pic2_dmc7 pic7_dmc7;
output out=mobile_lin_predict p=linear_predictions;
quit;


/* To plot ROC curve based on predictions from linear model */
proc logistic data=mobile_lin_predict plots=roc(id=prob);
 model install (event='1')= wifi device_height publisher_id_class2 publisher_id_class3 publisher_id_class5 publisher_id_class7 publisher_id_class8 wifi_pic7 dh_pic2 dh_pic5 dh_pic7 device_os_class3 device_os_class9 pic7_doc3 device_make_class7 device_make_class10 wifi_dmc7 pic2_dmc7 pic7_dmc7 /nofit;
roc pred=linear_predictions;
run;

/*ROC for logit model*/
proc logistic data=mobile_tran_with_indicators ;
 logit:  model install (event='1')= wifi device_height publisher_id_class1 publisher_id_class2 publisher_id_class3 wifi*publisher_id_class3 publisher_id_class5 publisher_id_class6 publisher_id_class7 wifi*publisher_id_class7 device_height*publisher_id_class7 publisher_id_class8 device_os_class3 publisher_id_class7*device_os_class3 device_os_class9 device_make_class7; 
  score data=mobile_test_with_indicators out=mobile_logit_predict;
run;
proc logistic data=mobile_logit_predict plots=roc(id=prob);
model install (event='1')= wifi device_height publisher_id_class1 publisher_id_class2 publisher_id_class3 wifi*publisher_id_class3 publisher_id_class5 publisher_id_class6 publisher_id_class7 wifi*publisher_id_class7 device_height*publisher_id_class7 publisher_id_class8 device_os_class3 publisher_id_class7*device_os_class3 device_os_class9 device_make_class7/nofit; 
 roc pred=p_1;
run;

/* ROC for oversampled data*/
proc logistic data=over_sample_tran;
 /*logit:  model model install (event='1')= wifi device_height publisher_id_class1 publisher_id_class2 publisher_id_class3 wifi*publisher_id_class3 publisher_id_class5 publisher_id_class6 publisher_id_class7 wifi*publisher_id_class7 device_height*publisher_id_class7 publisher_id_class8 device_os_class3 publisher_id_class7*device_os_class3 device_os_class9 device_make_class7;*/
logit: model install (event='1')= wifi resolution publisher_id_class2 resolution*publisher_id_class2 publisher_id_class3 publisher_id_class6 publisher_id_class10 device_os_class1  publisher_id_class6*device_os_class1 device_make_class3 device_make_class7 wifi*device_make_class7 device_make_class8 device_make_class10 device_os_class1*device_make_class10 ; 
weight w;/* score data=mobile_test_with_indicators out=mobile_logit_predict;*/
 score data=mobile_main out=mobile_logit_predict_oversample;
output out=out p=pwt;
run;

proc logistic data=mobile_logit_predict_oversample plots=roc(id=prob);
 /*model install (event='1')= wifi device_height publisher_id_class1 publisher_id_class2 publisher_id_class3 wifi*publisher_id_class3 publisher_id_class5 publisher_id_class6 publisher_id_class7 wifi*publisher_id_class7 device_height*publisher_id_class7 publisher_id_class8 device_os_class3 publisher_id_class7*device_os_class3 device_os_class9 device_make_class7 /nofit;*/
 model install (event='1')= wifi resolution publisher_id_class2 resolution*publisher_id_class2 publisher_id_class3 publisher_id_class6 publisher_id_class10 device_os_class1  publisher_id_class6*device_os_class1 device_make_class3 device_make_class7 wifi*device_make_class7 device_make_class8 device_make_class10 device_os_class1*device_make_class10 /nofit; 
roc pred=p_1;
run;

/**************************************Part 2 ************************************/

/*Linear probability model*/



data mobile_lin_predict;
set mobile_lin_predict;
if linear_predictions >= 0.001  then Thres_001 = 1 ;else Thres_001 = 0;
if linear_predictions >= 0.005 then Thres_005 = 1 ;else Thres_005 = 0;
if linear_predictions >= 0.010 then Thres_010 = 1 ;else Thres_010 = 0;
if linear_predictions >= 0.015 then Thres_015 = 1 ;else Thres_015 = 0;
if linear_predictions >= 0.020 then Thres_020 = 1 ;else Thres_020 = 0;
if linear_predictions >= 0.025 then Thres_025 = 1 ;else Thres_025 = 0;
if linear_predictions >= 0.030 then Thres_030 = 1 ;else Thres_030 = 0;
if linear_predictions >= 0.035 then Thres_035 = 1 ;else Thres_035 = 0;
if linear_predictions >= 0.040 then Thres_040 = 1 ;else Thres_040 = 0;
if linear_predictions >= 0.045 then Thres_045 = 1 ;else Thres_045 = 0;
if linear_predictions >= 0.050 then Thres_050 = 1 ;else Thres_050 = 0;
run;

data mobile_lin_predict_final;
set mobile_lin_predict;
if install = 0 AND  Thres_001 = 1  then pred_001=1 ;else if install = 1 AND  Thres_001 = 0 then pred_001 = 0;
if install = 0 AND  Thres_005 = 1  then pred_005=1 ;else if install = 1 AND  Thres_005 = 0 then pred_005 = 0;
if install = 0 AND  Thres_010 = 1  then pred_010=1 ;else if install = 1 AND  Thres_010 = 0 then pred_010 = 0;
if install = 0 AND  Thres_015 = 1  then pred_015=1 ;else if install = 1 AND  Thres_015 = 0 then pred_015 = 0;
if install = 0 AND  Thres_020 = 1  then pred_020=1 ;else if install = 1 AND  Thres_020 = 0 then pred_020 = 0;
if install = 0 AND  Thres_025 = 1  then pred_025=1 ;else if install = 1 AND  Thres_025 = 0 then pred_025 = 0;
if install = 0 AND  Thres_030 = 1  then pred_030=1 ;else if install = 1 AND  Thres_030 = 0 then pred_030 = 0;
if install = 0 AND  Thres_035 = 1  then pred_035=1 ;else if install = 1 AND  Thres_035 = 0 then pred_035 = 0;
if install = 0 AND  Thres_040 = 1  then pred_040=1 ;else if install = 1 AND  Thres_040 = 0 then pred_040 = 0;
if install = 0 AND  Thres_045 = 1  then pred_045=1 ;else if install = 1 AND  Thres_045 = 0 then pred_045 = 0;
if install = 0 AND  Thres_050 = 1  then pred_050=1 ;else if install = 1 AND  Thres_050 = 0 then pred_050 = 0;
run;


proc freq data = mobile_lin_predict_final;
tables pred_001 pred_005 pred_010 pred_015 pred_020 pred_025 pred_030 pred_035 pred_040 pred_045 pred_050/ out=freq_table;
run;

data dummy;
input Threshold $  FN FP;
datalines;
pred_001 0 119640 
pred_005 78 99587 
pred_010 652 23154
pred_015 866 7261
pred_020 984 976 
pred_025 1006 116
pred_030 1008 0
pred_035 1008 0
pred_040 1008 0
pred_045 1008 0
pred_050 1008 0
;

data dummy;
set dummy;
total_cost = FN*100 +FP*1;
sub=1;
run;


proc sgpanel  data= dummy;
 panelby sub;
 series x=total_cost y=Threshold;
run;

/*Total cost for logistic model*/
proc logistic data=mobile_tran_with_indicators outmodel=Logitmodel;
 logit:  model install (event='1')= wifi device_height publisher_id_class1 publisher_id_class2 publisher_id_class3 wifi*publisher_id_class3 publisher_id_class5 publisher_id_class6 publisher_id_class7 wifi*publisher_id_class7 device_height*publisher_id_class7 publisher_id_class8 device_os_class3 publisher_id_class7*device_os_class3 device_os_class9 device_make_class7; 
run;

proc logistic inmodel=Logitmodel;
 score data=mobile_test_with_indicators outroc=mobile_logit_roc;
run;


data mobile_logit_roc;
set mobile_logit_roc;
total_cost = _FALPOS_ * 1 + _FALNEG_ * 100;
run;


proc sort data = mobile_logit_roc;
by total_cost;
run;

proc sort data=mobile_logit_roc;
  by _PROB_;

data mobile_logit_roc;
set mobile_logit_roc;
sub=1;
run;
proc sgpanel  data= mobile_logit_roc;
panelby sub;
 series x=total_cost y=_PROB_;
run;

/*Total cost for the oversampled data*/


proc logistic data=over_sample_tran outmodel=Logitmodel_oversampled;
 logit:  model install (event='1')= wifi resolution publisher_id_class2 resolution*publisher_id_class2 publisher_id_class3 publisher_id_class6 publisher_id_class10 device_os_class1  publisher_id_class6*device_os_class1 device_make_class3 device_make_class7 wifi*device_make_class7 device_make_class8 device_make_class10 device_os_class1*device_make_class10 ; 
weight w;
run;

proc logistic inmodel=Logitmodel_oversampled;
 score data=mobile_main outroc=mobile_sampled_logit_roc;
run;


data mobile_sampled_logit_roc;
set mobile_sampled_logit_roc;
total_cost = _FALPOS_ * 1 + _FALNEG_ * 100;
run;


proc sort data = mobile_sampled_logit_roc;
by total_cost;
run;

data mobile_sampled_logit_roc;
set mobile_sampled_logit_roc;
sub=1;
run;
proc sgpanel  data= mobile_sampled_logit_roc;
panelby sub;
 series x=total_cost y=_PROB_;
run; 


