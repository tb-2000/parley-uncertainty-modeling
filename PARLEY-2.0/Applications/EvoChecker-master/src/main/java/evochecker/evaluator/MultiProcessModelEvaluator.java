//==============================================================================
//	
//	Copyright (c) 2020-
//	Authors:
//	* Simos Gerasimou (University of York)
//  * Faisal Alhwikem (University of York)
//	
//------------------------------------------------------------------------------
//	
//	This file is part of EvoChecker.
//	
//==============================================================================
package evochecker.evaluator;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;

import evochecker.auxiliary.Constants;
import evochecker.auxiliary.Utility;
import evochecker.genetic.problem.GeneticModelProblem;
import evochecker.genetic.problem.GeneticProblem;
import jmetal.core.Problem;
import jmetal.core.Solution;

/**
 * Class representing a parallel evaluator
 * @author sgerasimou
 *
 */
public class MultiProcessModelEvaluator implements IParallelEvaluator {
	/** number of parallel executions (processes)*/
	private int numberOfProcesses;

	private Problem[] problems;
	
	/** List of solutions*/
	private List<Solution> solutionsList;

	/** Solution results list*/
	private CopyOnWriteArrayList<Solution> evaluatedSolutions;

	/** Array of threads*/
	private Thread[] threads;

	/** Array of runnables*/
	private RunnableExecutor[] runnables;
		
	/** Set of connections array keeping the evaluators instances*/
	private Connection connections[];

	/** Number of completed parallel evaluation batches. */
	private int evaluationBatchCounter = 0;

	/** Restart PrismExecutor workers periodically to release native PRISM memory. */
	private static final int RESTART_EVERY_BATCHES = 1;

	
	/**
	 * Constructor
	 * @param processes
	 * @throws Exception 
	 */
	public MultiProcessModelEvaluator(){
		String processesNum = Utility.getProperty(Constants.PROCESSORS_KEYWORD);
		if (processesNum!=null)
			numberOfProcesses = Integer.parseInt(processesNum);
		else if (processesNum == null || processesNum.equals("-1"))
			numberOfProcesses = Runtime.getRuntime().availableProcessors();
			
		//initialise connections and executors
		int initPort = Integer.parseInt(Utility.getProperty(Constants.INITIAL_PORT_KEYWORD));
		connections = new Connection[numberOfProcesses];
		for (int i = 0; i < numberOfProcesses; i++) {
			try {
				connections[i] = new Connection(initPort + i, i, this);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}		
		
		//initialise threads, runnables and solutions
		threads 		= new Thread[numberOfProcesses];
		runnables 		= new RunnableExecutor[numberOfProcesses];
		solutionsList 	= new ArrayList<Solution>();
	}

	
	/** 
	 * Initialise evaluator 
	 */
	public void startEvaluator(Problem problem) {
		//System.out.println("Cores: " + numberOfProcesses);

		try {		
			problems = new Problem[numberOfProcesses];
			if (problem instanceof GeneticProblem){
				for (int i=0; i<numberOfProcesses; i++){
					problems[i] = problem;//new GeneticProblem((GeneticProblem) problem);
				}
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}
	

	/** 
	 * Add the solution to the list of solutions to be evaluated
	 */
	public void addSolutionForEvaluation(Solution solution) {
//		 System.out.println("Adding a solution to be evaluated");
		solutionsList.add(solution);
	}

	
	/**
	 * Run parallel evaluation
	 */
	public List<Solution> parallelEvaluation() {
//		System.out.println("Parallel evaluation");
		evaluatedSolutions = new CopyOnWriteArrayList<Solution>();
		this.reset();
		this.assignSolutions();
		this.startThreads();

		// All current model checks are complete here.
		solutionsList.clear();
		evaluationBatchCounter++;

		if (evaluationBatchCounter % RESTART_EVERY_BATCHES == 0) {
			System.out.println("Restarting PrismExecutor workers after evaluation batch " + evaluationBatchCounter);
			this.restartConnections();
		}

//		System.out.println("End of parallel evaluation....");
		return this.evaluatedSolutions;
	}
	
	
	/**
	* Assign solutions to parallel processes
	*/
	private void assignSolutions() {
		for (int i = 0; i < this.solutionsList.size(); i++) {
//			System.out.println("Assigning tasks");
			this.runnables[i % this.runnables.length].addSolutionForEvaluation(this.solutionsList
					.get(i));
		}
	}
	

	/**
	 * When done, reset the evaluators
	 */
	private void reset() {	
		for (int i = 0; i < numberOfProcesses; i++) {
			runnables[i] = new RunnableExecutor(connections[i], problems[i]); 
//					new RunnableExecutor(connections[i].getOutChannel(), connections[i].getInChannel(), problems[i]);
			threads[i] 	 = new Thread(runnables[i]);
		}
	}


	/**
	 * Start parallel execution
	 */
	private void startThreads() {
		for (Thread t : this.threads) {
			t.start();
		}

		for (Thread t : this.threads) {
			try {
				t.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

	
	/**
	 * Once finished, stop the evaluators
	 */
	public void stopEvaluator() {
		try {
			for (Connection c: connections) {
				c.close();
			}
		} 
		catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	
	

	/**
	 * Restart all PrismExecutor processes without restarting EvoChecker/NSGA-II.
	 * This is only called after all current evaluation threads have completed.
	 */
	private void restartConnections() {
		for (int i = 0; i < numberOfProcesses; i++) {
			try {
				int port = connections[i].getPort();
				connections[i].close();
				connections[i] = new Connection(port, i, this);
				System.out.println("Restarted PrismExecutor worker " + i + " on port " + port);
			}
			catch (Exception e) {
				e.printStackTrace();
				throw new RuntimeException("Failed to restart PrismExecutor worker " + i, e);
			}
		}
	}

	public void setConnection(int id, Connection c) {
		connections[id] = c;
	}
	
	/**
	 * Inner class
	 * @author sgerasimou
	 *
	 */
	private class RunnableExecutor implements Runnable {
		/** List of solutions to be evaluated*/
		private List<Solution> solutionsList = new ArrayList<Solution>();

 		/** Output*/
		private PrintWriter out;

		/** Input*/
		private BufferedReader in;

		/** Problem to be handled by this executor*/
		Problem runnableProblem;
		
		Connection connection;
		
		
		/**
		 * Class constructor: create a new runnable executor
		 * @param out
		 * @param in
		 */
		public RunnableExecutor(PrintWriter out, BufferedReader in, Problem problem) {
			this.in 			 = in;
			this.out 			 = out;
			this.runnableProblem = problem;
			this.solutionsList 	 = new ArrayList<Solution>();
		}
		
		
		public RunnableExecutor(Connection c, Problem problem) {
			this.in 			 = c.getInChannel();
			this.out 			 = c.getOutChannel();
			this.runnableProblem = problem;
			this.solutionsList 	 = new ArrayList<Solution>();
			this.connection		 = c; 
		}


		/** Add a solution for evaluation*/
		public void addSolutionForEvaluation(Solution solution) {
			this.solutionsList.add(solution);
		}
		

		/**
		 * Run
		 */
		@Override
		public void run() {
			for (Solution solution : this.solutionsList) {
				try {
					if (runnableProblem instanceof GeneticModelProblem){
						boolean OK = ((GeneticModelProblem) runnableProblem).parallelEvaluate(in, out, solution);
						if (!OK) {
							this.connection = new Connection(connection);
							this.in			= connection.getInChannel();
							this.out		= connection.getOutChannel();
						}	
					}
					else throw new IllegalArgumentException("Problem not recognised");
				} catch (Exception e) {
					e.printStackTrace();
				}
//				 System.out.println("Adding result");
				evaluatedSolutions.add(solution);
			}
		}
	}
	
	
	
	private class Connection {

		/** Socket **/
		private Socket socket;
		
		/** Input channel**/
		private BufferedReader in;
		
		/** Output channel**/
		private PrintWriter out;

		/** PrismExecutor process owned by this connection. */
		private Process process;

		private final String HOSTNAME = "127.0.0.1";
		
		private int portNum;
		
		private MultiProcessModelEvaluator evaluator;
		
		private int id;

		public Connection(int portNum, int id, MultiProcessModelEvaluator evaluator) throws Exception {
			this.portNum 	= portNum;
			this.evaluator 	= evaluator;
			this.id			= id;
			start();
		}
		
		public Connection (Connection c) throws Exception {
			this(c.portNum, c.id, c.evaluator);
			evaluator.setConnection(id, this);
		}
		
		
		public void start() {
			String params[] = new String[4];
			params[0] = Utility.getProperty(Constants.JAVA_KEYWORD);
			params[1] = "-jar";
			params[2] = Utility.getProperty(Constants.MODEL_CHECKING_ENGINE);
			params[3] = String.valueOf(portNum);

			System.out.println("Starting PrismExecutor worker " + id + " on port " + portNum);

			try {
				ProcessBuilder pb = new ProcessBuilder(params);
				Map<String, String> env = pb.environment();
				env.put("DYLD_LIBRARY_PATH", Utility.getProperty(Constants.MODEL_CHECKING_ENGINE_LIBS_DIR));
				env.put("LD_LIBRARY_PATH", Utility.getProperty(Constants.MODEL_CHECKING_ENGINE_LIBS_DIR));

				// Start exactly one worker and keep the Process reference.
				process = pb.start();

				boolean successful = false;
				while (!successful) {
					if (!process.isAlive()) {
						throw new IOException("PrismExecutor worker " + id + " terminated before opening port " + portNum);
					}

					try {
						socket = new Socket(HOSTNAME, portNum);
						System.out.println("Connected worker " + id + " to port " + portNum);
						in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
						out = new PrintWriter(socket.getOutputStream());
						successful = true;
					}
					catch (IOException | NullPointerException e) {
						// Worker may still be starting. Do not spawn another worker here.
						Thread.sleep(1000);
					}
				}
			}
			catch (IOException | InterruptedException e) {
				if (e instanceof InterruptedException) {
					Thread.currentThread().interrupt();
				}
				throw new RuntimeException("Failed to start PrismExecutor worker " + id + " on port " + portNum, e);
			}
		}

		public BufferedReader getInChannel() {
			return in;
		}

		
		public PrintWriter getOutChannel() {
			return out;
		}
		
		protected int getPort() {
			return portNum;
		}

		
		public void close() throws IOException {
			if (out != null) {
				out.close();
			}
			if (in != null) {
				in.close();
			}
			if (socket != null && !socket.isClosed()) {
				socket.close();
			}

			if (process != null && process.isAlive()) {
				process.destroy();
				try {
					if (!process.waitFor(5, TimeUnit.SECONDS)) {
						process.destroyForcibly();
						process.waitFor(5, TimeUnit.SECONDS);
					}
				}
				catch (InterruptedException e) {
					Thread.currentThread().interrupt();
					process.destroyForcibly();
				}
			}
		}
	}


}
