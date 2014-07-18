package dut.ir.CRFJava;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import riso.numerical.LBFGS;

/**
 * FileName： CRFTrainer.java <br/>
 * package：dut.ir.CRFJava <br/>
 * author：zmz <br/>   
 * date：2014-7-11 <br/>
 * time：下午8:46:42 <br/>
 * email: zmz@mail.dlut.edu.cn <br/>
 * 功能：训练CRF参数，并将训练得到的参数写入文件 <br/>
 *     CRF的理论推导可参考李航老师的《统计学习方法》 <br/>
 *     CRFJava使用LBFGS来优化目标函数，使用RISO Project中的LBFGS.java文件，感谢作者的工作。<br/><br/>
 *     
 *     如发现问题，请发送email至 zmz@mail.dlut.edu.cn，谢谢！
 */

public class CRFTrainer extends AbstractCRF {
	/**
	 * delta * delta，负对数似然函数与权重的调和参数，该值越大，越拟合训练数据
	 */
	private double delta2;
	
	/**
	 * 状态特征的个数
	 */
	private int uFeatureNum;
	/**
	 * 转移特征的函数
	 */
	private int bFeatureNum;
	
	public CRFTrainer(String trainFile, String templateFile, String modelFile,
			int threshold, double delta) throws IOException {
		this.delta2 = delta * delta;
		this.modelFile = modelFile;
		crfParams = new CRFParams();
		// 初始化templates
		initTemplates(templateFile);
		System.out.println("模板：" + crfParams.templates.size());

		// 初始化：sentences，sentenceLabels，labels
		initSentencesAndLabels(trainFile);
		crfParams.K = crfParams.labels.size();
		System.out.println("句子：" + sentences.size());
		System.out.println("标签数：" + crfParams.K);

		// 初始化：crfParams.uFeatureAddr，crfParams.bFeatureAddr，uFunWeight,bFunWeight,
		initFeatureInfo(threshold);
		System.out.println("状态特征：" + crfParams.uFeatureAddr.size());
		System.out.println("转移特征：" + crfParams.bFeatureAddr.size());
		System.out.println("特征函数：" + (crfParams.bFeatureAddr.size() * crfParams.K * crfParams.K + crfParams.uFeatureAddr.size()*crfParams.K));

		//初始化：uFeatureArray,bFeatureArray
		initFeatureArray(this.sentences, this.sentenceLabels);
		// 初始化：uFeatureArray，bFeatureArray，uFeatureStart，uFeatureEnd，bFeatureStart，bFeatureEnd
		initWordFeatureInfo(this.sentences, this.sentenceLabels);
	}
	
	public static void main(String[] args) throws IOException {
		CRFTrainer crf = new CRFTrainer("data/train.data", "data/template", "data/model",1, 1);
		crf.train();
		//MyTool.showMap(crfParams.uFeatureAddr);
	}
	
	/**
	 * 使用LBFGS算法进行训练
	 */
	public void train(){
		double  diag1[];

		double f, eps, xtol, gtol, t1, t2, stpmin, stpmax;
		int iprint[], iflag[] = new int[1], iter, n, m, mp, lp, j;
		iprint = new int[2];
		boolean diagco;
		iprint[0] = -1;
		iprint[1] = 0;
		diagco = false;
		eps = 1.0e-10;
		xtol = 1.0e-16;
		iter = 0;
		iflag[0] = 0;
		m = 7;
		diag1 = new double[crfParams.funWeight.length];
		
		double preLoss = 0;//前一次的损失值
		double relLoss = 0;//与前一次损失值的相对差别
		
		do {
			double[] gradient = new double[crfParams.funWeight.length];
			double loss = loss(gradient);
			double[] p = this.checkAllaccuracy();
			if(iter == 0){
				relLoss = 1;
			}
			else{
				relLoss = (preLoss - loss) / preLoss;
			}
			preLoss = loss;
			System.out.printf("iter=%d  loss=%f  句子准确率：%f  单词准确率:%f  相对损失:%f\n",iter,loss,p[0],p[1],relLoss);
			//System.out.println("iter=" + iter + "\tloss=" + loss + "\t句子准确率：" + p[0] + "\t单词准确率：" + p[1]);

			try {
				LBFGS.lbfgs(crfParams.funWeight.length, m, crfParams.funWeight, loss, gradient, diagco, diag1, iprint, eps, xtol,
						iflag);
				
			} catch (LBFGS.ExceptionWithIflag e) {
				System.err.println("Sdrive: lbfgs failed.\n" + e);
				return;
			}
			
			iter += 1;
		} while (iflag[0] != 0 && iter <= 100 && relLoss > 0.00001);
		
		this.writeModel();
	}
	
	/**
	 * 计算句子级别和单词级别的准确率
	 * @return 句子级别的准确率和单词级别的准确率
	 */
	private double[] checkAllaccuracy(){
		double[] p = new double[2];
		int corS = 0;//正确的句子数
		int corT  = 0;//正确的单词数
		int tokens = 0;//所有的单词数
		boolean right = true;
		
		for(int i = 0; i < sentences.size(); i++){
			int[] tags = new int[sentences.get(i).length];
			Integer[] sLabels = sentenceLabels.get(i);
			int[] pLabels = this.viterbi(i);//预测值
			right = true;//句子是否完全正确
			
			for(int j = 0; j < pLabels.length; j++){
				if(sLabels[j] == pLabels[j]){
					corT++;
				}
				else{
					right = false;
				}
			}
			tokens += pLabels.length;
		}
		p[0] = 1.0 * corS / sentences.size();
		p[1] = 1.0 * corT / tokens;
		return p;
	}

	/**
	 * 根据当前参数，计算训练损失函数的值，并计算参数的梯度向量
	 * @param gradient 梯度向量
	 * @return 损失函数的值
	 */
	public double loss(double[] gradient) {
		int iter = 0;
		Integer[] uStart;
		Integer[] uEnd;
		Integer[] bStart;
		Integer[] bEnd;
		int addr;

		double loss = 0;
		
		for (int s = 0; s < sentences.size(); s++) {// 遍历所有句子
			uStart = uFeatureStart.get(s);
			uEnd = uFeatureEnd.get(s);
			bStart = bFeatureStart.get(s);
			bEnd = bFeatureStart.get(s);

			ArrayList<double[][]> Ms = this.computeMs(s);
			Integer[] sLabels = sentenceLabels.get(s);
			double[][] alpha = this.alpha(Ms);
			double[][] beta = this.beta(Ms);
			double zeta1 = this.zetaByAlpha(alpha);
			
			if(Double.isInfinite(zeta1)//如果句子太长，归一化参数容易变为无穷大，所以句子应该尽可能的短 
				continue;
				
			loss += Math.log(zeta1);

			for (int i = 0; i < Ms.size(); i++) {// 遍历每个单词
				double[][] M = Ms.get(i);

				for(int j = uStart[i]; j >=0 && j <= uEnd[i]; j++){//更新状态特征函数的梯度
					addr = uFeatureArray[j];//特征索引
					loss = loss - crfParams.funWeight[addr + sLabels[i]];
					gradient[addr + + sLabels[i]] -= 1;
					
					for(int k = 0; k < crfParams.K; k++){//addr处，第k个特征函数
						gradient[addr + k] += alpha[k][i]*beta[k][i] / zeta1;
						//gradient[addr + k] += crfParams.funWeight[addr + k] * alpha[k][i]*beta[k][i] / zeta1;
					}
				}
				if(i > 0){
					for(int j = bStart[i]; j >=0 && j <= bEnd[i]; j++){//更新转移特征函数的梯度
						addr = bFeatureArray[j];//特征索引
						loss -= crfParams.funWeight[addr + sLabels[i-1] * crfParams.K + sLabels[i]];
						gradient[addr + sLabels[i-1] * crfParams.K + sLabels[i]] -= 1;
						
						for(int preK = 0; preK < crfParams.K; preK++){
							for(int k = 0; k < crfParams.K;k++){
								gradient[addr + preK * crfParams.K + k] +=  alpha[preK][i-1]*beta[k][i] * M[preK][k] / zeta1;
								//gradient[addr + preK * crfParams.K + k] += crfParams.funWeight[addr + preK * crfParams.K + k] * alpha[preK][i-1]*beta[k][i] * M[preK][k] / zeta1;
							}
						}
					}
				}
			}
		}// 遍历句子完成

		for(int k = 0; k < crfParams.funWeight.length; k++){
			loss += 0.5 * crfParams.funWeight[k] * crfParams.funWeight[k] / delta2;
			gradient[k] = gradient[k] + crfParams.funWeight[k] / delta2;
		}
		return loss;
	}
	
	/**
	 * 通过alpha矩阵计算zeta
	 * @param alpha alpha矩阵
	 * @return 归一化因子zeta
	 */
	public double zetaByAlpha(double[][] alpha) {
		double zeta = 0;
		int length = alpha[0].length;

		for (int i = 0; i < crfParams.K; i++) {
			zeta += alpha[i][length - 1];
		}
		return zeta;
	}
	
	/**
	 * 计算前向后向算法中的alpha矩阵
	 * 
	 * @param Ms 句子每个单词的M矩阵组成的列表
	 * @return alpha矩阵
	 */
	public double[][] alpha(ArrayList<double[][]> Ms) {
		double[][] alpha = new double[crfParams.K][Ms.size()];

		for (int i = 0; i < Ms.size(); i++) {// 依此求取每个位置的alpha
			double[][] M = Ms.get(i);// i处的M矩阵
			if (i == 0) {
				for (int k = 0; k < crfParams.K; k++) {
					alpha[k][i] = M[0][k];
				}
			} else {
				for (int k = 0; k < crfParams.K; k++) {
					for (int preK = 0; preK < crfParams.K; preK++) {
						alpha[k][i] += alpha[preK][i - 1] * M[preK][k];
					}
				}
			}
		}
		return alpha;
	}
	
	/**
	 * 计算前向后向算法中的beta矩阵
	 * @param Ms 句子每个单词的M矩阵组成的列表
	 * @return beta矩阵
	 */
	public double[][] beta(ArrayList<double[][]> Ms) {
		double[][] beta = new double[crfParams.K][Ms.size()];
		for (int i = Ms.size() - 1; i >= 0; i--) {
			if (i == Ms.size() - 1) {
				for (int k = 0; k < crfParams.K; k++)
					beta[k][i] = 1;
			} else {
				double[][] M = Ms.get(i + 1);
				for (int k = 0; k < crfParams.K; k++) {
					for (int nextK = 0; nextK < crfParams.K; nextK++) {
						beta[k][i] += beta[nextK][i + 1] * M[k][nextK];
					}
				}
			}
		}
		return beta;
	}
	
	
	/**
	 * 将crfParmas写入文件
	 */
	private void writeModel(){
		FileOutputStream fos;
		ObjectOutputStream oos;
		try {
			fos = new FileOutputStream(this.modelFile);
			oos = new ObjectOutputStream(fos);
			oos.writeObject(crfParams);
			oos.close();
			
		}catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	/**
	 * 根据当前sentences中存储的句子，初始化：crfParams.uFeatureAddr，crfParams.bFeatureAddr，crfParams.funWeight<br/>
	 * 
	 * @param threshold 特征频次阈值，出现次数小于该阈值的特征被删除
	 */
	public void initFeatureInfo(int threshold) {
		// 首先：crfParams.uFeatureAddr和crfParams.bFeatureAddr中存储每个特征出现的次数，
		crfParams.uFeatureAddr = new HashMap<String, Integer>();
		crfParams.bFeatureAddr = new HashMap<String, Integer>();
		String feature;

		// Entry<String,Integer[]> temps = templates.;
		for (int i = 0; i < sentences.size(); i++) {
			String[][] sentence = sentences.get(i);
			for (int m = 0; m < sentence.length; m++) {
				Iterator<Entry<String, Integer[]>> it = crfParams.templates.entrySet()
						.iterator();
				while (it.hasNext()) {
					Entry<String, Integer[]> entry = it.next();
					feature = entry.getKey() + ":";
					Integer[] numbers = entry.getValue();

					for (int k = 0; k < numbers.length; k = k + 2) {
						if (numbers[k] + m < 0|| numbers[k] + m >= sentence.length)
							continue;
						feature += sentence[numbers[k] + m][numbers[k + 1]];
					}

					if (feature.startsWith("U")) {// 状态特征模板
						if (crfParams.uFeatureAddr.containsKey(feature))
							crfParams.uFeatureAddr.put(feature,
									1 + crfParams.uFeatureAddr.get(feature));
						else
							crfParams.uFeatureAddr.put(feature, 1);
					} else {// 转移特征模板
						if (crfParams.bFeatureAddr.containsKey(feature))
							crfParams.bFeatureAddr.put(feature,
									1 + crfParams.bFeatureAddr.get(feature));
						else
							crfParams.bFeatureAddr.put(feature, 1);
					}
				}
			}
		}

		// 精简特征
		Iterator<String> it = crfParams.uFeatureAddr.keySet().iterator();
		while (it.hasNext()) {
			feature = it.next();
			if (crfParams.uFeatureAddr.get(feature) < threshold) {
				it.remove();
				crfParams.uFeatureAddr.remove(feature);
			}
		}
		it = crfParams.bFeatureAddr.keySet().iterator();
		while (it.hasNext()) {
			feature = it.next();
			if (crfParams.bFeatureAddr.get(feature) <= threshold) {
				it.remove();
				crfParams.bFeatureAddr.remove(feature);
			}
		}

		uFeatureNum = crfParams.uFeatureAddr.size();
		bFeatureNum = crfParams.bFeatureAddr.size();

		crfParams.funWeight = new double[uFeatureNum * crfParams.K + bFeatureNum * crfParams.K * crfParams.K];

		it = crfParams.uFeatureAddr.keySet().iterator();
		int index = 0;
		while (it.hasNext()) {
			crfParams.uFeatureAddr.put(it.next(), index);
			index = index + crfParams.K;
		}
		
		it = crfParams.bFeatureAddr.keySet().iterator();
		while (it.hasNext()) {
			crfParams.bFeatureAddr.put(it.next(), index);
			index = index + crfParams.K * crfParams.K;
		}
	}
	
	
	/**
	 * 读取所有训练句子，并初始化每个句子对应的标签数组
	 * @param trainFile
	 */
	public void initSentencesAndLabels(String trainFile){
		sentences = new ArrayList<String[][]>();
		sentenceLabels = new ArrayList<Integer[]>();
		crfParams.labels = new ArrayList<String>();
		try {
			BufferedReader in = new BufferedReader(new FileReader(trainFile));
			while(true){
				String[][] tem = this.readSingleSentence(in);
				if(tem.length == 0)
					break;
				sentences.add(tem);
				sentenceLabels.add(this.mapSentenceLabelToNum(tem));
			}
			in.close();
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		//return sentences;
	}
	
	
	/**
	 * 初始化模板列表，模板格式详见CRFParams.templates
	 * @param templateFile
	 * @throws FileNotFoundException
	 */
	private void initTemplates(String templateFile) throws IOException {
		BufferedReader in = new BufferedReader(new FileReader(templateFile));
		crfParams.templates = new HashMap<String, Integer[]>();
		String line = "";
		int row = -1;// 行号
		int col = -1; // 列号

		Pattern p = Pattern.compile("\\[(-*\\d+),(\\d+)\\]");// 匹配[-2,1]
		while (null != (line = in.readLine())) {
			if (line.startsWith("U") || line.startsWith("B")) {
				String[] tem = line.split(":");
				ArrayList<Integer> numbers = new ArrayList<Integer>();

				if (tem.length == 2) {
					Matcher m = p.matcher(tem[1]);
					while (m.find()) {
						row = Integer.valueOf(m.group(1));
						col = Integer.valueOf(m.group(2));
						numbers.add(row);
						numbers.add(col);
					}
					crfParams.templates.put(tem[0], numbers.toArray(new Integer[0]));
				}

				else {// 特征模板: B
					crfParams.templates.put(tem[0], new Integer[0]);
				}
			}
		}
		in.close();
	}

}
