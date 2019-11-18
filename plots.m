Y = winequalitywhite{ :,end};
X1_fixedacid = winequalitywhite{ :,1};
X2_volatileacidity = winequalitywhite{ :,2};
X3_criticacid = winequalitywhite{ :,3};
X4_residualsugar = winequalitywhite{ :,4};
X5_chlorides = winequalitywhite{ :,5};
X6_freesulfurdioxide = winequalitywhite{ :,6};
X7_totalsulfurdioxide = winequalitywhite{ :,7};
X8_density = winequalitywhite{ :,8};
X9_pH = winequalitywhite{:,9};
X10_sulphates = winequalitywhite{ :,10};
X11_alcohol = winequalitywhite{ :,11};

hold on;
for i = 1:11
    figure(i);
    scatter(winequalitywhite{:,i}, Y);
end

