
imgfolder = '/scratch/16824/data/crop_imgs/';
testfile = 'joint_results.txt';
gtfile = '/home/xiaolonw/assignment/data/testlist_both.txt';


fid = fopen(testfile, 'r');
fid2= fopen(gtfile, 'r');

overlaps = [];


cnt = 0; 
accuracy = 0; 
while ~feof(fid)

	s = fscanf(fid, '%s', 1); 
	if length(s) ==0 
		break;
	end
	cnt = cnt + 1;
	s2 = fscanf(fid2, '%s', 1);
	predlbl = fscanf(fid, '%d', 1);
	gtlbl = fscanf(fid2, '%d', 1);
	bb = fscanf(fid, '%f', 4);
	bbgt = fscanf(fid2, '%f', 4);
	ov = 0;

	if predlbl == gtlbl 
		accuracy = accuracy + 1;
	end

	if mod(cnt, 1000) == 0
		fname = [imgfolder s];
		im = imread(fname);
		bbox = int32(bb); 
		bbox2 = int32(bbgt); 
		im = drawbbox(im, [bbox(2), bbox(4), bbox(1), bbox(3)], 2,  [255,0,0] );
		im = drawbbox(im, [bbox2(2), bbox2(4), bbox2(1), bbox2(3)], 2,  [0,0,255] );
		imwrite(im, ['imgs/' num2str(cnt) '.jpg']);
	end

    bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
    iw=bi(3)-bi(1)+1;
    ih=bi(4)-bi(2)+1;
    if iw>0 & ih>0
        % compute overlap as area of intersection / area of union
        ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
           (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
           iw*ih;
        ov=iw*ih/ua;
    end

    overlaps = [overlaps; ov];

end


fclose(fid);
fclose(fid2);

prop1 = size(find(overlaps(:) > 0.5), 1) / size(overlaps, 1);
prop2 = size(find(overlaps(:) > 0.7), 1) / size(overlaps, 1);
prop3 = size(find(overlaps(:) > 0.8), 1) / size(overlaps, 1);

accuracy = accuracy / cnt; 
fprintf('Accuracy: %f, IOU 0.5: %f, IOU 0.7: %f IOU 0.8: %f', accuracy, prop1, prop2, prop3);








