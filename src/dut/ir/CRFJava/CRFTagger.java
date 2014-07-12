package dut.ir.CRFJava;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.ObjectInputStream;
import java.util.ArrayList;

/**
 * FileName： CRFTagger.java <br/>
 * package：dut.ir.CRFJava <br/>
 * author：zmz <br/>   
 * date：2014-7-11 <br/>
 * time：下午9:09:06 <br/>
 * email: zmz@mail.dlut.edu.cn <br/>
 * 功能：TODO <br/>
 */

public class CRFTagger extends AbstractCRF{

	/**
	 * 加载模型文件
	 * @param modelFile 模型文件
	 * @throws Exception
	 */
	public CRFTagger(String modelFile) throws Exception{
		initCRFParams(modelFile);
	}
	
	
	public static void main(String[] args) throws Exception{
		CRFTagger tagger = new CRFTagger("data/model");
		tagger.tagger("data/test.data", "data/out.txt");
	}

	/**
	 * 对一个文件进行标记，依次读取所有句子，对于每个句子进行标记，并将结果写入文件
	 * @param testFile 待标记的文件
	 * @param outFile 标记结果输出文件
	 * @throws Exception
	 */
	public void tagger(String testFile,String outFile) throws Exception{
		BufferedReader in = new BufferedReader(new FileReader(testFile));
		BufferedWriter out = new BufferedWriter(new FileWriter(outFile));
		
		while(true){
			String[][] sentence = this.readSingleSentence(in);
			Integer[] sLabels = new Integer[sentence.length];
			
			if(sentence.length == 0){
				in.close();
				out.close();
				return;
			}
			
			this.initContext(sentence, sLabels);
			int[] pLabels = this.viterbi(0);
			
			for(int i = 0; i < sLabels.length; i++){
				for(int j = 0; j < crfParams.col; j++){
					out.write(sentence[i][j] + "\t");
				}
				out.write(crfParams.labels.get(pLabels[i]));
				out.newLine();
			}
			out.newLine();
		}
	}
	
	
	/**
	 * 根据传递的单个句子和相应的标签数组，实例化父类的成员变量，便于计算所有的M矩阵，以及进行viterbi运算
	 * @param sentence
	 * @param sLabels
	 * @throws Exception
	 */
	private void initContext(String[][] sentence, Integer[] sLabels) throws Exception{
		this.sentences = new ArrayList<String[][]>();
		this.sentenceLabels = new ArrayList<Integer[]>();
		this.sentenceLabels.add(sLabels);
		this.sentences.add(sentence);
		
		this.initFeatureArray(this.sentences, this.sentenceLabels);
		this.initWordFeatureInfo(this.sentences, this.sentenceLabels);
	}
	
	/**
	 * 从模型文件中读取CRFParmas变量，式礼服父类中的crfParams
	 * @param modelFile
	 * @throws Exception
	 */
	private void initCRFParams(String modelFile) throws Exception{
		FileInputStream fis=new FileInputStream(modelFile);
		ObjectInputStream ois=new ObjectInputStream(fis);
		
		this.crfParams = (CRFParams) ois.readObject();
		ois.close();
	}
}
