package dut.ir.CRFJava;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map.Entry;

/**
 * FileName： AbstractCRF.java <br/>
 * package：dut.ir.CRFJava <br/>
 * author：zmz and yy<br/>   
 * date：2014-7-11 <br/>
 * time：下午5:06:02 <br/>
 * email: zmz@mail.dlut.edu.cn <br/>
 * 功能：抽象类， CRF的训练和标记过程有许多相同的操作和运算，比如都需要计算句子在每个位置的M矩阵，都需要进行viterbi运算，所以定义一个抽象类，作为CRFTrainer和CRFTagger的父类 <br/>
 */

public abstract class AbstractCRF {
	/**
	 * 采用与CRF++相同的数据格式，具体参见：<a>http://crfpp.googlecode.com/svn/trunk/doc/index.html#usage</a>
	 * 每个句子用二维数组表示，包含标签列 <br/>
	 * 在CRFTrainer中，其包含所有的句子 <br/>
	 * 在CRFTagger中，只包含一个句子，标记完成后再加载另一个句子<br/>
	 */
	public ArrayList<String[][]> sentences;
	
	/**
	 * sentences中每个句子对应的标签数组，注：标签使用其索引表示<br/>
	 * 与sentences相对应
	 */
	public ArrayList<Integer[]> sentenceLabels;
	

	/**
	 * 把所有句子所有单词的状态特征依次存入该数组，状态特征使用其在CRFParams.funWeight数组中的首地址表示
	 */
	public int uFeatureArray[];
	
	/**
	 * 把所有句子所有单词的转移特征依次存入该数组，转移特征使用其在CRFParams.funWeight数组中的首地址表示
	 */
	public int bFeatureArray[];
	
	/**
	 * sentences中每个句子对应一个数组，记录句子中每个单词所有的状态特征在uFeatureArray中的起始地址
	 */
	public ArrayList<Integer[]> uFeatureStart;
	/**
	 * sentences中每个句子对应一个数组，记录句子中每个单词所有的状态特征在uFeatureArray中的结束地址，与uFeatureStart对应
	 */
	public ArrayList<Integer[]> uFeatureEnd;
	
	/**
	 * sentences中每个句子对应一个数组，记录句子中每个单词所有的转移特征在bFeatureArray中的起始地址，同uFeatureStart
	 */
	public ArrayList<Integer[]> bFeatureStart;
	/**
	 * sentences中每个句子对应一个数组，记录句子中每个单词所有的转移特征在bFeatureArray中的结束地址，与bFeatureStart对应，同bFeatureEnd
	 */
	public ArrayList<Integer[]> bFeatureEnd;// 同bFeatureEnd
	
	/**
	 * 所有单词的状态特征总数，也就是uFeatureArray的大小
	 */
	public int uFeatureCount = 0;
	
	/**
	 * 所有单词的庄毅特征总数，也就是bFeatureArray的大小
	 */
	public int bFeatureCount = 0;
	
	/**
	 * 模型文件 <br/>
	 * CRFTrainer将训练得到的CRFParams实例写入该文件  <br/>
	 * CRFTagger从该文件中读取训练得到的CRFParmas实例
	 */
	public String modelFile;
	
	/**
	 * CRF模型参数
	 */
	public CRFParams crfParams;
	
	public AbstractCRF(){
		
	}
	
	/**
	 * 根据当前参数，标记句子
	 * @param sid 句子在sentences数组中的索引
	 * @return 标记数组
	 */
	public int[] viterbi(int sid){
		int[] sLabels = new int[sentences.get(sid).length];
		
		int[][] tag = new int[crfParams.K][sentences.get(sid).length];
		double[][] delta = new double[crfParams.K][sentences.get(sid).length];
		ArrayList<double[][]> Ms = this.computeMs(sid);
		double maxP = -1;
		int cur = -1;
		
		for(int j = 0; j < sentences.get(sid).length; j++){
			double[][] M = Ms.get(j);
			for(int i = 0; i < crfParams.K; i++){
				if(j == 0){
					tag[i][j] = -1;
					delta[i][j] = M[0][i];
				}
				else{
					maxP = -1;
					cur = -1;
					for(int k = 0; k < crfParams.K; k++){
						if(delta[k][j-1] * M[k][i] > maxP){
							maxP = delta[k][j-1] * M[k][i];
							cur = k;
						}
					}
					delta[i][j] = maxP;
					tag[i][j] = cur;
				}
			}
		}
		int maxK = -1;
		maxP = -1;
		for(int k = 0; k < crfParams.K;k++){
			if(delta[k][sentences.get(sid).length-1] > maxP){
				maxP = delta[k][sentences.get(sid).length-1];
				maxK = k;
			}
		}
		
		for(int i = sentences.get(sid).length-1; i >= 0; i--){
			if(i == sentences.get(sid).length-1){
				sLabels[sentences.get(sid).length-1] = maxK;
			}
			else{
				sLabels[i] = tag[maxK][i+1];
				maxK = sLabels[i];
			}
		}
		
		return sLabels;
	}
	
	
	/**
	 * 计算句子在每个位置的M矩阵 <br/> 在每个位置，M是K * K的矩阵
	 * 
	 * @param sid
	 * @return 返回所有M矩阵组成的列表
	 */
	public ArrayList<double[][]> computeMs(int sid) {
		ArrayList<double[][]> Ms = new ArrayList<double[][]>();

		Integer[] uStart = uFeatureStart.get(sid);
		Integer[] uEnd = uFeatureEnd.get(sid);
		Integer[] bStart = bFeatureStart.get(sid);
		Integer[] bEnd = bFeatureEnd.get(sid);

		double state = 0;// 在位置i处，状态特征的加权和
		double trans = 0;// 在位置i出，转移特征的加权和
		int addr = 0;// 该特征所对应的特征函数列表的起始地址

		for (int i = 0; i < sentences.get(sid).length; i++) {// 依此遍历每个单词
			double[][] M = new double[crfParams.K][crfParams.K];
			if (i == 0) {// 第一个单词位置，单独处理
				for (int n = 0; n < crfParams.K; n++) {
					state = this.sumOfuFunctions(uStart, uEnd, i, n);
					M[0][n] = Math.exp(state);
				}
				Ms.add(M);
			} else {
				for (int n = 0; n < crfParams.K; n++) {// 当前位置的标签序号
					state = this.sumOfuFunctions(uStart, uEnd, i, n);
					for (int m = 0; m < crfParams.K; m++) {
						trans = this.sumOfbFuntions(bStart, bEnd, i, m, n);

						M[m][n] = Math.exp(state + trans);
					}
				}
				Ms.add(M);
			}
		}
		return Ms;
	}
	
	/**
	 * 求某个句子第i个单词出满足的所有状态特征函数的权值之和
	 * @param uStart 句子在uFeatureStart中的项，记录每个单词在uFeatureArray中的起始地址
	 * @param uEnd 句子在uFeatureEnd中的项，记录每个单词在uFeatureArray中的结束地址
	 * @param i 第i个单词
	 * @param curLabel 当前的标记
	 * @return 句子第i个单词出满足的所有状态特征函数的权值之和
	 */
	public double sumOfuFunctions(Integer[] uStart, Integer[] uEnd, int i,
			int curLabel) {
		double result = 0;
		int addr = 0;
		for (int k = uStart[i]; k >= 0 && k <= uEnd[i]; k++) {
			addr = uFeatureArray[k] + curLabel;
			result += crfParams.funWeight[addr];
		}

		return result;
	}
	
	/**
	 * 求某个句子第i个单词出满足的所有转移特征函数的权值之和
	 * @param bStart 句子在bFeatureStart中的项，记录每个单词在bFeatureArray中的起始地址
	 * @param bEnd 句子在bFeatureEnd中的项，记录每个单词在bFeatureArray中的结束地址
	 * @param i 第i个单词
	 * @param preLabel 第i-1个单词的标记
	 * @param curLabel 当前的标记
	 * @return 求某个句子第i个单词出满足的所有转移特征函数的权值之和
	 */
	public double sumOfbFuntions(Integer[] bStart, Integer[] bEnd, int i,
			int preLabel, int curLabel) {
		double result = 0;
		int addr = 0;
		for (int k = bStart[i]; k >= 0 && k <= bEnd[i]; k++) {
			addr = bFeatureArray[k] + crfParams.K * preLabel + curLabel;
			result += crfParams.funWeight[addr];
		}
		return result;
	}
	
	/**
	 * 生成uFeatureArray和bFeatureArray后，初始化uFeatureStart，uFeatureEnd，bFeatureStart和bFeatureEnd <br/>
	 * 如果某个位置没有状态或者转移特征，则记为-1
	 * @param allSentences
	 * @param allSentencesLabels
	 */
	public void initWordFeatureInfo(ArrayList<String[][]> allSentences,ArrayList<Integer[]> allSentencesLabels) {
		uFeatureStart = new ArrayList<Integer[]>();
		uFeatureEnd = new ArrayList<Integer[]>();
		bFeatureStart = new ArrayList<Integer[]>();
		bFeatureEnd = new ArrayList<Integer[]>();

		String feature;
		int uIndex = 0;// uFeatureArray的索引
		int bIndex = 0;//

		int uStartIndex = -1;
		int bStartIndex = -1;

		for (int i = 0; i < allSentences.size(); i++) {// 遍历每个句子
			String[][] sentence = allSentences.get(i);// 句子
			Integer[] curLabels = allSentencesLabels.get(i);// 当前句子的标签

			Integer[] uStart = new Integer[curLabels.length];
			Integer[] uEnd = new Integer[curLabels.length];
			Integer[] bStart = new Integer[curLabels.length];
			Integer[] bEnd = new Integer[curLabels.length];

			for (int m = 0; m < sentence.length; m++) {// 遍历每个单词
				uStartIndex = -1;
				bStartIndex = -1;

				Iterator<Entry<String, Integer[]>> it = crfParams.templates.entrySet()
						.iterator();
				while (it.hasNext()) {// 遍历每个模板
					Entry<String, Integer[]> entry = it.next();
					feature = entry.getKey() + ":";
					Integer[] numbers = entry.getValue();

					for (int k = 0; k < numbers.length; k = k + 2) {
						if (numbers[k] + m < 0
								|| numbers[k] + m >= sentence.length)
							continue;
						feature += sentence[numbers[k] + m][numbers[k + 1]];
					}

					if (feature.startsWith("U")
							&& crfParams.uFeatureAddr.containsKey(feature)) {
						uFeatureArray[uIndex] = crfParams.uFeatureAddr.get(feature);
						if (uStartIndex == -1)
							uStartIndex = uIndex;
						uIndex++;
					} else if (crfParams.bFeatureAddr.containsKey(feature) && m > 0) {
						bFeatureArray[bIndex] = crfParams.bFeatureAddr.get(feature);
						if (bStartIndex == -1)
							bStartIndex = bIndex;
						bIndex++;
					}
				}

				uStart[m] = uStartIndex;
				uEnd[m] = (uStartIndex == -1) ? -1 : uIndex - 1;
				bStart[m] = bStartIndex;
				bEnd[m] = (bStartIndex == -1) ? -1 : bIndex - 1;
			}
			uFeatureStart.add(uStart);
			uFeatureEnd.add(uEnd);
			bFeatureStart.add(bStart);
			bFeatureEnd.add(bEnd);
		}
	}
	
	/**
	 * 根据当前所有的句子和标签信息初始化uFeatureArray和bFeatureArray数组 <br/>
	 * CRFTrainer： 利用所有训练句子和训练句子的标签来初始化uFeatureArray和bFeatureArray数组 <br/>
	 * CRFTagger： 对于每个要标记的句子及其“标签”（可以为占位符），初始化uFeatureArray和bFeatureArray数组， <br/>
	 * @param allSentences 当前的句子列表
	 * @param allSentencesLabels 当前句子标签列表
	 */
	public void initFeatureArray(ArrayList<String[][]> allSentences,ArrayList<Integer[]> allSentencesLabels) {
		String feature;
		uFeatureCount = 0;
		bFeatureCount = 0;

		for (int i = 0; i < allSentences.size(); i++) {// 遍历每个句子
			String[][] sentence = allSentences.get(i);// 句子
			Integer[] curLabels = allSentencesLabels.get(i);// 当前句子的标签
			for (int m = 0; m < sentence.length; m++) {// 遍历每个单词
				Iterator<Entry<String, Integer[]>> it = crfParams.templates.entrySet()
						.iterator();
				while (it.hasNext()) {// 遍历每个模板
					Entry<String, Integer[]> entry = it.next();
					feature = entry.getKey() + ":";
					Integer[] numbers = entry.getValue();

					for (int k = 0; k < numbers.length; k = k + 2) {
						if (numbers[k] + m < 0
								|| numbers[k] + m >= sentence.length)
							continue;
						feature += sentence[numbers[k] + m][numbers[k + 1]];
					}

					if (feature.startsWith("U")
							&& crfParams.uFeatureAddr.containsKey(feature)) {// 状态特征
						//int labelIndex = curLabels[m];
						uFeatureCount++;

					} else if (crfParams.bFeatureAddr.containsKey(feature) && m > 0) {
						bFeatureCount++;
					}
				}
			}
		}
		uFeatureArray = new int[uFeatureCount];
		bFeatureArray = new int[bFeatureCount];
	}
	
	/**
	 * 将一个句子的标签用其序号代替，并返回标签序号数组
	 * @param sentence
	 * @return 句子对应的标签数组
	 */
	public Integer[] mapSentenceLabelToNum(String[][] sentence){
		Integer[] sLabels = new Integer[sentence.length];
		for(int i = 0; i < sLabels.length; i++){
			if (crfParams.labels.contains(sentence[i][crfParams.col - 1])) {
				sLabels[i] = crfParams.labels.indexOf(sentence[i][crfParams.col - 1]);
				
			} else {
				crfParams.labels.add(sentence[i][crfParams.col - 1]);
				sLabels[i] = crfParams.labels.indexOf(sentence[i][crfParams.col - 1]);
			}
		}
		return sLabels;
	}
	
	/**
	 * 从文件中读取一个句子（每个句子以空行表示结束）
	 * @param in 文件流
	 * @return 句子的二维数组
	 * @throws IOException
	 */
	public String[][] readSingleSentence(BufferedReader in) throws IOException{
		String line = "";
		ArrayList<String[]> temWords = new ArrayList<String[]>();
		while(true){
			line = in.readLine();
			if(line == null || line.length() == 0){
				break;
			}
			String[] token = line.split("\\s");
			//初始化列数
			if(crfParams.col == 0)
				crfParams.col = token.length;
			//判断
			if(token.length != crfParams.col){
				System.err.println("文件格式不正确：列数不一致");
				System.err.println(line);
				return null;
			}
			temWords.add(token);
		}
		String[][] tokens = new String[temWords.size()][crfParams.col];
		for(int i = 0; i < temWords.size(); i++){
			for(int j = 0; j < crfParams.col; j++){
				tokens[i][j] = temWords.get(i)[j];
			}
		}
		return tokens;
	}
	
	
}



























